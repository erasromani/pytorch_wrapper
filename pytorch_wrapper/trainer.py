import os
import gin
import dataclasses
import numpy as np
import logging
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Any, Callable, List, Dict
from collections import defaultdict

from pytorch_wrapper.metrics import MetricConfig
from pytorch_wrapper.utils import repr_torchdict

logger = logging.getLogger(__name__)

@gin.configurable
@dataclasses.dataclass
class Trainer:
    optimizer: Any
    output_dir: str
    max_epochs: int
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_metrics: List[str] = dataclasses.field(default_factory=dict)
    checkpoint_monitor: Optional[str] = None
    eval_freq: Optional[int] = None
    tb_writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None

    def __post_init__(self):
        self.eval_metrics = MetricConfig(self.eval_metrics).get_functions()
        if self.checkpoint_monitor is not None:
            assert self.checkpoint_monitor in self.eval_metrics.keys(), "invalid checkpoint_monitor value {} entered".format(self.checkpoint_monitor)

    @gin.configurable("Trainer.fit")
    def fit(self, model, datamodule, log_interval=None):
        self.iter = 0
        train_loader = datamodule.train_dataloader()
        valid_loader = datamodule.valid_dataloader()
        
        model.to(self.device)
        self.checkpoint_metric = -np.Inf
        for epoch in range(1, self.max_epochs + 1):
            self.epoch = epoch
            self.train(train_loader, model, log_interval=log_interval, valid_loader=valid_loader)
            if self.eval_freq is None:
                metrics = self.validate(valid_loader, model)
                self.save_checkpoint(model, metrics)

        self.tb_writer.flush()
        self.tb_writer.close()

    def save_checkpoint(self, model, metrics):
        if self.checkpoint_monitor is not None and metrics[self.checkpoint_monitor] > self.checkpoint_metric:
            self.checkpoint_metric = metrics[self.checkpoint_monitor]
            checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'iter': self.iter,
                'model_config': model.config,
                self.checkpoint_monitor: metrics[self.checkpoint_monitor]
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info('\nSaved model to ' + checkpoint_path + '.')

    def train(self, train_loader, model, log_interval=None, valid_loader=None):
        if self.eval_freq is not None:
            assert valid_loader is not None, "valid_loader must be passed as a keyword argument"
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input']
            target = batch['target']
            for i, input in enumerate(inputs):
                if isinstance(input, torch.Tensor):
                    inputs[i] = input.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = model(*inputs)
            loss = self.loss_function(output, target).mean()
            loss.backward()
            self.optimizer.step()
            if log_interval is not None and batch_idx % log_interval == 0 and batch_idx > 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(target), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
                
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('loss/train', loss, self.iter)
                self.tb_writer.add_scalar('epoch', self.epoch, self.iter)
            
            if self.eval_freq is not None and self.iter % self.eval_freq == 0 and self.iter > 0:
                metrics = self.validate(valid_loader, model)
                self.save_checkpoint(model, metrics)

            self.iter += 1

    @torch.no_grad()
    def validate(self, valid_loader, model):
        model.eval()
        validation_loss = 0
        numels = defaultdict(int)
        metrics = defaultdict(int)
        for batch in valid_loader:
            inputs = batch['input']
            target = batch['target']
            for i, input in enumerate(inputs):
                if isinstance(input, torch.Tensor):
                    inputs[i] = input.to(self.device)
            target = target.to(self.device)
            output = model(*inputs)
            loss = self.loss_function(output, target)
            numels['loss'] = loss.numel()
            validation_loss += loss.sum()
            for metric_name, eval_metric in self.eval_metrics.items():
                metric = eval_metric(output, target)
                numels[metric_name] += metric.numel()
                metrics[metric_name] += metric.sum()

        validation_loss /= numels['loss']
        for key, value in metrics.items():
            metrics[key] = value / numels[key]

        logger.info('\nValidation set: Average loss: {:.4f}, {}\n'.format(
            validation_loss, repr_torchdict(metrics)))
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('loss/val', validation_loss, self.iter)
            for key, value in metrics.items():
                self.tb_writer.add_scalar('{}/val'.format(key), value, self.iter)
        return metrics
