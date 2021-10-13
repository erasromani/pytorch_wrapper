import torch
import dataclasses
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Any, Callable

@dataclasses.dataclass
@gin.configurable
class DataModule(Dataset):
    batch_size: int
    train_dataset: torch.utils.data.dataset.Dataset
    valid_dataset: torch.utils.data.dataset.Dataset
    test_dataset: torch.utils.data.dataset.Dataset
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
    
    def get_loaders(self):
        return self.train_dataloader(), self.valid_dataloader(), self.test_dataloader()