import dataclasses
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Any

from pytorch_wrapper.dataset import DataModule


@dataclasses.dataclass
class Trainer:
    optimizer: Any
    output_dir: str
    max_epochs: int
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tb_writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None

    def fit(self, model, datamodule, log_interval=10):
        self.iter = 0
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        model.to(self.device)
        max_metric = -np.Inf
        for epoch in range(1, self.max_epochs + 1):
            self.epoch = epoch
            self.train(train_loader, model, log_interval=log_interval)
            metric = self.validation(val_loader, model)
            if metric > max_metric:
                max_metric = metric
                checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'model_config': model.config,
                    'accuracy': metric
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
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('loss/train', loss, self.iter)
                self.tb_writer.add_scalar('epoch', self.epoch, self.iter)
            self.iter += 1


    @torch.no_grad()
    def validation(self, val_loader, model):
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        metric = correct / len(val_loader.dataset)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('loss/val', validation_loss, self.iter)
            self.tb_writer.add_scalar('accuracy/val', metric, self.iter)
        return metric
