import gin
import torch
import dataclasses
import torch.nn.functional as F

from typing import Any, Callable, List

def accuracy(input, target):
    input = F.softmax(input, dim=1)
    pred = input.max(1, keepdim=True)[1]
    accuracy = pred.eq(target.view_as(pred))
    return accuracy

@gin.configurable
@dataclasses.dataclass
class MetricConfig:
    names: List[Callable[[Any, Any], Any]]
        
    def get_metric_func(self, name):
        if name == "accuracy":
            return accuracy
        else:
            raise ValueError("invalid name {}".format(name))
        
    def get_functions(self):
        functions = {}
        for name in self.names:
            functions[name] = self.get_metric_func(name)
        return functions