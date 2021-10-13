import torch
import random
import math
import torch.nn.functional as F


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
