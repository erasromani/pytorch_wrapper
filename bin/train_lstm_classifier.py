import gin
import os
import numpy as np
import torch

from torchtext.legacy import data
from torchtext.legacy import datasets
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from pytorch_wrapper.models.nlp_models import LSTMClassifier
from pytorch_wrapper.data.processing import NLPPipeline
from pytorch_wrapper.data import DataModule
from pytorch_wrapper.optimizer import OptimConfig
from pytorch_wrapper.trainer import Trainer
from pytorch_wrapper import DATA_DIR
from pytorch_wrapper.utils import load_text_data, set_seed, get_device, gin_wrap
from pytorch_wrapper.callbacks.callbacks import Callback


class IMDBRawDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        text = " ".join(self.data[index].text)
        label = self.data[index].label
        if label == 'pos':
            label = 1
        elif label == 'neg':
            label = 0
        else:
            raise ValueError("invalid label text {}".format(label))
        return text, label

    def __len__(self):
        return len(self.data) 

    def process_data(self):
        text = []
        label = []
        for sample_text, sample_label in self:
            text.append(sample_text)
            label.append(sample_label)
        return text, label

@gin.configurable
class IMDBDataset(Dataset):
    def __init__(self, text, label, max_seq_length=300, padding_idx=0):
        self.text = text
        self.label = label
        self.max_seq_length = max_seq_length
        self.padding_idx = padding_idx
        assert len(text) == len(label), "text and label must be the same length"
    
    def __getitem__(self, index):
        text = self.text[index][:self.max_seq_length]
        label = self.label[index]

        # Pad
        arr = np.pad(text, (0, self.max_seq_length - len(text)), constant_values=self.padding_idx)
        assert len(arr) == self.max_seq_length
        return arr, label
    
    def __len__(self):
        return len(self.label)


@gin.configurable
def get_datamodule():
    TEXT = data.Field(fix_length=500)
    LABEL = data.LabelField(dtype = torch.long)

    train_data, valid_data = datasets.IMDB.splits(TEXT, LABEL)
    train_ds = IMDBRawDataset(train_data)
    valid_ds = IMDBRawDataset(valid_data)

    train_data, train_label = train_ds.process_data()
    valid_data, valid_label = valid_ds.process_data()

    nlp = NLPPipeline()
    train_data_indices, valid_data_indices = nlp(train_data, valid_data)

    train_ds = IMDBDataset(train_data_indices, train_label)
    valid_ds = IMDBDataset(valid_data_indices, valid_label)
    dm = DataModule(train_dataset=train_ds, valid_dataset=valid_ds)
    return dm, nlp

@gin.configurable
def start_experiment(output_dir, max_epochs, seed=None):
    set_seed(seed=seed)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tb_logs'))
    device = get_device()
    dm, nlp = get_datamodule()

    model = LSTMClassifier(input_dim=len(nlp.id2token), padding_idx=nlp.padding_idx)
    optimizer = OptimConfig().create_optimizer(model)
    trainer = Trainer(optimizer, output_dir, max_epochs, tb_writer=writer, device=device)

    trainer.fit(model, dm)
    

if __name__ == "__main__":

    gin_wrap(start_experiment)