import gin
import dataclasses
import torch
from typing import Optional


@dataclasses.dataclass
@gin.configurable
class OptimConfig:
    optimizer: str
    learning_rate: float
    weight_decay: Optional[float] = None

    def create_optimizer(self, model):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("invalid optimizer {}".format(self.optimizer))
        
        return optimizer
