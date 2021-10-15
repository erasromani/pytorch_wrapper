import gin
import torch
import dataclasses
import logging

from torch.utils.data import Dataset, DataLoader
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)

@gin.configurable
@dataclasses.dataclass
class DataModule(Dataset):
    batch_size: int
    train_dataset: torch.utils.data.dataset.Dataset
    valid_dataset: torch.utils.data.dataset.Dataset
    test_dataset: Optional[torch.utils.data.dataset.Dataset] = None
    num_workers: int = 4
    collate_fn: Optional[Callable[[Any], Any]] = None
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
    
    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        if self.test_dataset is None:
            logger.warning("no test_dataset given")
            return None
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                            shuffle=False, num_workers=self.num_workers,
                            collate_fn=self.collate_fn)
    
    def get_loaders(self):
        if self.test_data is not None:
            return self.train_dataloader(), self.valid_dataloader(), self.test_dataloader()
        else:
            return self.train_dataloader(), self.valid_dataloader()
