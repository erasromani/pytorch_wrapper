import os
import torch
import numpy as np
import logging
import gin
import random
import dataclasses

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from pytorch_wrapper import DATA_DIR
from pytorch_wrapper.data import DataModule
from pytorch_wrapper.models.vision_models import ModelConfig, ImageClassificationNet
from pytorch_wrapper.trainer import Trainer
from pytorch_wrapper.utils import get_output_dir, gin_wrap, set_seed
from pytorch_wrapper.optimizer import OptimConfig
from pytorch_wrapper.data.augmentation import TransformModule

logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):
        self.X = torch.load(X_path).squeeze(1)
        self.y = torch.load(y_path).squeeze(1)
        self.transform = transform if transform is not None else lambda x: x
    
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]


def get_datamodule():
        transform = TransformModule()
        train_ds = MyDataset(X_path=os.path.join(DATA_DIR, 'train/X.pt'), y_path=os.path.join(DATA_DIR, 'train/y.pt'), transform=transform.train)
        valid_ds = MyDataset(X_path=os.path.join(DATA_DIR, 'validation/X.pt'), y_path=os.path.join(DATA_DIR, 'validation/y.pt'), transform=transform.val)
        dm = DataModule(train_dataset=train_ds, valid_dataset=valid_ds)
        return dm

@gin.configurable
def start_experiment(output_dir, max_epochs, seed=None):
    set_seed(seed=seed)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tb_logs'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Running on {}".format(device))

    dm = get_datamodule()
    model_config = ModelConfig()
    model = ImageClassificationNet(model_config)
    optimizer = OptimConfig().create_optimizer(model)
    trainer = Trainer(optimizer, output_dir, max_epochs, tb_writer=writer, device=device)

    trainer.fit(model, dm)


if __name__ == "__main__":

    gin_wrap(start_experiment)
