import dataclasses
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Any, Callable, List, Dict
from collections import defaultdict


@dataclasses.dataclass
class Trainer:
    optimizer: Any
    output_dir: str
    max_epochs: int
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluation_functions: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = dataclasses.field(default_factory=dict)
    checkpoint_monitor: Optional[str] = None
    valid_freq: Optional[int] = None
    tb_writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None

    def __post_init__(self):
        if self.checkpoint_monitor is not None:
            assert self.checkpoint_monitor in self.evluation_functions.keys(), "invalid checkpoint_monitor value {} entered".format(self.checkpoint_monitor)

    def fit(self, model, datamodule, log_interval=10):
        self.iter = 0
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        model.to(self.device)
        max_metric = -np.Inf
        for epoch in range(1, self.max_epochs + 1):
            self.epoch = epoch
            self.train(train_loader, model, log_interval=log_interval)
            metrics = self.validate(val_loader, model)
            if self.checkpoint_monitor is not None and metrics[self.checkpoint_monitor] > max_metric:
                max_metric = metrics[self.checkpoint_monitor]
                checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'model_config': model.config,
                    self.checkpoint_monitor: metrics[self.checkpoint_monitor]
                }
                torch.save(checkpoint, checkpoint_path)
                print('\nSaved model to ' + checkpoint_path + '.')

        self.tb_writer.flush()
        self.tb_writer.close()

    def train(self, train_loader, model, log_interval=10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.loss_function(output, target).mean()
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
                
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('loss/train', loss, self.iter)
                self.tb_writer.add_scalar('epoch', self.epoch, self.iter)
            self.iter += 1


    @torch.no_grad()
    def validate(self, val_loader, model):
        model.eval()
        validation_loss = 0
        numels = defaultdict(int)
        metrics = defaultdict(int)
        for data, target in val_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = model(data)
            loss = self.loss_function(output, target)
            numels['loss'] = loss.numel()
            validation_loss += loss.sum()
            for metric_name, evaluation_function in self.evaluation_functions.items():
                metric = evaluation_function(output, target)
                numels[metric_name] += metric.numel()
                metrics[metric_name] += metric.sum()

        validation_loss /= numels['loss']
        for key, value in metrics.item():
            metrics[key] = value / numels[key]

        print('\nValidation set: Average loss: {:.4f}, {}\n'.format(
            validation_loss.item(), metrics))
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('loss/val', validation_loss, self.iter)
            for key, value in metrics.item():
                self.tb_writer.add_scalar('{}/val'.format(key), value, self.iter)
        return metrics
