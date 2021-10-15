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
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    callbacks: List[Any] = dataclasses.field(default_factory=list)
    checkpoint_monitor: Optional[str] = None
    eval_freq: Optional[int] = None
    tb_writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None

    def __post_init__(self):
        self.training = None
    
    @gin.configurable("Trainer.fit")
    def fit(self, model, datamodule, log_interval=None):
        assert hasattr(model, 'training_step'), "model must have training_step attribute"
        assert hasattr(model, 'validation_step'), "model must have validation_step attribute"
        self.iter = 0
        train_loader = datamodule.train_dataloader()
        valid_loader = datamodule.valid_dataloader()
        self.on_fit_start(model)

        model.to(self.device)
        self.checkpoint_metric = -np.Inf
        for self.epoch in range(1, self.max_epochs + 1):
            self.on_epoch_start(model)
            self.train(train_loader, model, log_interval=log_interval, valid_loader=valid_loader)
            if self.eval_freq is None:
                metrics = self.validate(valid_loader, model)
                self.save_checkpoint(model, metrics)
            self.on_epoch_end(model)

        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        self.on_fit_end(model)

    def save_checkpoint(self, model, metrics):
        if self.checkpoint_monitor is not None and metrics[self.checkpoint_monitor] > self.checkpoint_metric:
            self.checkpoint_metric = metrics[self.checkpoint_monitor]
            checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'iter': self.iter,
                self.checkpoint_monitor: metrics[self.checkpoint_monitor]
            }
            if hasattr(model, 'config'):
                checkpoint['model_config'] = model.config
            torch.save(checkpoint, checkpoint_path)
            logger.info('saved model to ' + checkpoint_path)
    
    def on_fit_start(self, model):
        model.on_fit_start(self)
        for cb in self.callbacks:
            cb.on_fit_start(self, model)

    def on_fit_end(self, model):
        model.on_fit_end(self)
        for cb in self.callbacks:
            cb.on_fit_end(self, model)

    def on_epoch_start(self, model):
        model.on_epoch_start(self)
        for cb in self.callbacks:
            cb.on_epoch_start(self, model)

    def on_epoch_end(self, model):
        model.on_epoch_end(self)
        for cb in self.callbacks:
            cb.on_epoch_end(self, model)

    def on_train_start(self, model):
        model.on_train_start(self)
        for cb in self.callbacks:
            cb.on_train_start(self, model)

    def on_train_end(self, model):
        model.on_train_end(self)
        for cb in self.callbacks:
            cb.on_train_end(self, model)

    def on_validation_start(self, model):
        model.on_validation_start(self)
        for cb in self.callbacks:
            cb.on_validation_start(self, model)

    def on_validation_end(self, model):
        metrics = model.on_validation_end(self)
        for cb in self.callbacks:
            cb.on_validation_end(self, model)
        return metrics

    def on_batch_start(self, model):
        model.on_batch_start(self)
        for cb in self.callbacks:
            cb.on_batch_start(self, model)

    def on_validation_batch_start(self, model, batch, batch_idx):
        model.on_validation_batch_start(self, batch, batch_idx)
        for cb in self.callbacks:
            cb.on_validation_batch_start(self, model, batch, batch_idx)

    def on_validation_batch_end(self, model, outputs, batch, batch_idx):
        model.on_validation_batch_end(self, outputs, batch, batch_idx)
        for cb in self.callbacks:
            cb.on_validation_batch_end(self, outputs, model, batch, batch_idx)

    def on_batch_end(self, model):
        model.on_batch_end(self)
        for cb in self.callbacks:
            cb.on_batch_end(self, model)

    def on_before_backward(self, model, loss):
        model.on_before_backward(self, loss)
        for cb in self.callbacks:
            cb.on_before_backward(self, model, loss)

    def on_after_backward(self, model):
        model.on_after_backward(self)
        for cb in self.callbacks:
            cb.on_after_backward(self, model)

    def on_before_optimizer_step(self, model):
        model.on_before_optimizer_step(self, self.optimizer)
        for cb in self.callbacks:
            cb.on_before_optimizer_step(self, model, self.optimizer)

    def on_before_zero_grad(self, model):
        model.on_before_zero_grad(self, self.optimizer)
        for cb in self.callbacks:
            cb.on_before_zero_grad(self, model, self.optimizer)

    def on_batch_end(self, model):
        model.on_batch_end(self)
        for cb in self.callbacks:
            cb.on_batch_end(self, model)

    def to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
        elif isinstance(batch, (tuple, list)):
            for index, value in enumerate(batch):
                if isinstance(value, torch.Tensor):
                    batch[index] = value.to(self.device)
        else:
            raise ValueError("invalid batch type {}".format(type(batch)))
        return batch

    def _extract_loss(self, output):
        if isinstance(output, torch.Tensor) and output.dim() == 0:
            if output.dim() == 0:
                output = {'loss': output}
            else:
                raise ValueError("loss must be a zero-dimensional tensor")
        elif isinstance(output, dict):
            assert hasattr(output, 'loss') and output['loss'].dim() == 0, "training_step hook must output a zero-dimensional loss tensor or a dictionary with keyword 'loss'"
        else:
            raise ValueError("invalid training_step output type {}".format(type(output)))
        loss = output['loss']
        return loss
        
    def train(self, train_loader, model, log_interval=None, valid_loader=None):
        self.training = True
        if self.eval_freq is not None:
            assert valid_loader is not None, "valid_loader must be passed as a keyword argument"
        model.train()
        self.on_train_start(model)
        for batch_idx, batch in enumerate(train_loader):
            batch = self.to_device(batch)
            self.on_batch_start(model)
            outputs = model.training_step(batch, batch_idx)
            loss = self._extract_loss(outputs)
            self.on_before_backward(model, loss)
            loss.backward()
            self.on_after_backward(model)
            self.on_before_optimizer_step(model)
            self.optimizer.step()
            self.on_before_zero_grad(model)
            self.optimizer.zero_grad()
            if log_interval is not None and batch_idx % log_interval == 0 and batch_idx > 0:
                logger.info('train: epoch={} ({:.0f}%), loss={:.6f}'.format(
                    self.epoch, 100. * batch_idx / len(train_loader), loss))
                
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('loss/train', loss, self.iter)
                self.tb_writer.add_scalar('epoch', self.epoch, self.iter)
            
            if self.eval_freq is not None and self.iter % self.eval_freq == 0 and self.iter > 0:
                metrics = self.validate(valid_loader, model)
                self.training = True
                model.train()
                self.save_checkpoint(model, metrics)

            self.on_batch_end(model)
            self.iter += 1
        self.on_train_end(model)
        self.training = None

    @torch.no_grad()
    def validate(self, valid_loader, model):
        self.training = False
        model.eval()
        self.on_validation_start(model)
        for batch_idx, batch in enumerate(valid_loader):
            self.on_validation_batch_start(model, batch, batch_idx)
            batch = self.to_device(batch)
            outputs = model.validation_step(batch, batch_idx)
            self.on_validation_batch_end(model, outputs, batch, batch_idx)

        metrics = self.on_validation_end(model)
        logger.info('val: epoch={}, {}'.format(self.epoch, repr_torchdict(metrics)))
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar('{}/val'.format(key), value, self.iter)

        self.training = None
        return metrics
