import math
import torch
import gin
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
from typing import Tuple, List

from pytorch_wrapper.models import nnModule

def conv_layer(in_channels, out_channels, kernel_size, stride=1):
    assert stride in (1, 2)
    layers = [
              nn.BatchNorm2d(in_channels),
              nn.ReLU(),
              ]
    if stride == 1:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', stride=stride))
    elif stride == 2:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=math.floor((kernel_size -1) / 2), stride=stride))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        assert stride in (1, 2)
        super(ResBlock, self).__init__()
        self.conv1 = conv_layer(in_channels, out_channels, kernel_size, stride=stride)
        self.conv2 = conv_layer(out_channels, out_channels, kernel_size, stride=1)
        if in_channels != out_channels:
            if stride == 1:
                self.ooconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', stride=stride)
            else:
                self.ooconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='valid', stride=stride)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if hasattr(self, 'ooconv'): x = self.ooconv(x)
        return x + out


class GlobalPooling(nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()
    
    def forward(self, x):
        out = x.mean(dim=-1).mean(dim=-1)
        return out


class SEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, r=16):
        assert stride in (1, 2)
        super(SEResBlock, self).__init__()
        self.resblock = ResBlock(in_channels, out_channels, kernel_size, stride=stride)
        self.senet = nn.Sequential(
            GlobalPooling(),
            nn.Linear(out_channels, out_channels // r),
            nn.ReLU(),
            nn.Linear(out_channels // r, out_channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        out = self.resblock.conv1(x)
        out = self.resblock.conv2(out)
        if hasattr(self.resblock, 'ooconv'): x = self.resblock.ooconv(x)
        seout = self.senet(out)
        return x + out * seout[..., None, None]


@gin.configurable
@dataclasses.dataclass
class ModelConfig:
    channels: List[int]
    kernels: List[int]
    strides: List[int]
    senet: List[bool] = False
    input_size: Tuple[int] = (3, 32, 32)
    num_hidden: int = 1000
    num_outputs: int = 43
    avg_pool: bool = True
    r: int = 16

    def __post_init__(self):
        if isinstance(self.channels, int): self.channels = [self.channels]
        if isinstance(self.strides, int):  self.strides  = [self.strides] * len(self.channels)
        if isinstance(self.kernels, int):  self.kernels  = [self.kernels] * len(self.channels)
        if isinstance(self.senet, bool):   self.senet    = [self.senet] * len(self.channels)
        assert len(self.strides) == len(self.kernels) == len(self.senet)


class ImageClassificationNet(nnModule):
    def __init__(self, config):
        super(ImageClassificationNet, self).__init__()
        self.config = config

        conv_layers = []
        if config.senet[0]:
            conv_layers.append(SEResBlock(config.input_size[0], config.channels[0], config.kernels[0], stride=config.strides[0], r=config.r))
        else:
            conv_layers.append(ResBlock(config.input_size[0], config.channels[0], config.kernels[0], stride=config.strides[0]))

        for i in range(1, len(config.channels)):
            if config.senet[i]:
                conv_layers.append(SEResBlock(config.channels[i - 1], config.channels[i], config.kernels[i], stride=config.strides[i], r=config.r))
            else:
                conv_layers.append(ResBlock(config.channels[i - 1], config.channels[i], config.kernels[i], stride=config.strides[i]))
        if config.avg_pool: conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        conv_layers.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_layers)
        with torch.no_grad():
            x = torch.rand(1, *config.input_size)
            out = self.conv(x)
        self.linear = nn.Sequential(
            nn.Linear(out.shape[-1], config.num_hidden),
            nn.ReLU(),
            nn.Linear(config.num_hidden, config.num_outputs),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return F.log_softmax(x,dim=1)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        return loss
            
    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target, reduction="sum")
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum()
        self.n_examples += len(data)
        return {'loss': loss, 'correct': correct}

    def on_validation_start(self, trainer):
        self.logs = []
        self.n_examples = 0

    def on_validation_batch_end(self, trainer, outputs, batch, batch_idx):
        self.logs.append(outputs)

    def on_validation_end(self, trainer):
        loss = torch.stack([log['loss'] for log in self.logs]).sum() / self.n_examples
        accuracy = torch.stack([log['correct'] for log in self.logs]).float().sum() / self.n_examples
        return {'loss': loss, 'accuracy': accuracy}
