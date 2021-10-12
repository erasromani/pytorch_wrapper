import os
import torch
import dataclasses
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Any, List

from pytorch_wrapper.data.augmentation import RotateImage


class MyDataset(Dataset):
    def __init__(self, X_path="X.pt", y_path="y.pt", transform=None):
        self.X = torch.load(X_path).squeeze(1)
        self.y = torch.load(y_path).squeeze(1)
        self.transform = transform if transform is not None else lambda x: x
    
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]


@dataclasses.dataclass
class TransformModule:
    train_augs: Optional[List[str]] = None
    val_augs:   Optional[List[str]] = None

    def __post_init__(self):
        self.train = self.get_transform(self.train_augs)
        self.val = self.get_transform(self.val_augs)
        
    def get_transform(self, augs):
        if augs is None:
            transform = None
        else:
            tfms = []
            for aug in augs:
                if aug == 'hflip':
                    tfms.append(transforms.RandomHorizontalFlip(p=0.5))
                elif aug == 'rotate':
                    tfms.append(RotateImage(10))
                elif aug == 'torch_rotate':
                    tfms.append(transforms.RandomRotation(10))
                elif aug == 'affine':
                    tfms.append(transforms.RandomAffine(degrees=(-15, 15), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)))
                else:
                    raise ValueError("invalid augmentation {} entered".format(aug))
            transform = transforms.Compose(tfms)
        return transform


@dataclasses.dataclass
class DataModule(Dataset):
    batch_size: int
    root: str = 'data/'
    num_workers: int = 4
    transform: TransformModule = TransformModule()

    def __post_init__(self):
        self.train_ds = MyDataset(X_path=os.path.join(self.root, 'train/X.pt'), y_path=os.path.join(self.root, 'train/y.pt'), transform=self.transform.train)
        self.val_ds = MyDataset(X_path=os.path.join(self.root, 'validation/X.pt'), y_path=os.path.join(self.root, 'validation/y.pt'), transform=self.transform.val)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
