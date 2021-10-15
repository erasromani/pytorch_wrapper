import torch
import gin
import random
import math
import torch.nn.functional as F
import dataclasses

from torchvision import transforms
from typing import Optional, List

class RotateImage:
    def __init__(self, max_rotate, padding_mode='reflection'):
        self.max_rotate = max_rotate
        self.padding_mode = padding_mode

    def __call__(self, img):
        theta = random.uniform(-self.max_rotate, self.max_rotate) * math.pi / 180
        M = self.rotation_matrix(theta)
        grid = F.affine_grid(M[:2][None, ...], img[None, ...].shape, align_corners=False)
        return F.grid_sample(img[None, ...], grid, padding_mode=self.padding_mode, align_corners = False).squeeze(0)
    
    def rotation_matrix(self, theta):
        M = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                          [math.sin(theta),  math.cos(theta), 0],
                          [0              ,  0              , 1]])
        return M


@gin.configurable
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
