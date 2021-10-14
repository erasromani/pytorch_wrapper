import os
import torch
import numpy as np
import logging
import gin
import random

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import fetch_20newsgroups

from pytorch_wrapper import DATA_DIR
from pytorch_wrapper.data import DataModule
from pytorch_wrapper.data.processing import NLPPipeline
from pytorch_wrapper.models.nlp_models import BagOfWords
from pytorch_wrapper.trainer import Trainer
from pytorch_wrapper.utils import get_output_dir, gin_wrap, set_seed
from pytorch_wrapper.optimizer import OptimConfig

logger = logging.getLogger(__name__)

@gin.configurable
def newsgroup_collate_func(batch, max_sentence_length=300):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    
    for datum in batch:
        # padding
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0, max_sentence_length-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return {"input": [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list)], "target": torch.LongTensor(label_list)}

@gin.configurable
def get_datamodule(train_split=10000, max_sentence_length=300):
    newsgroup_train = fetch_20newsgroups(subset='train', data_home=DATA_DIR)
    newsgroup_test = fetch_20newsgroups(subset='test', data_home=DATA_DIR)

    train_data = newsgroup_train.data[:train_split]
    train_targets = newsgroup_train.target[:train_split]

    valid_data = newsgroup_train.data[train_split:]
    valid_targets = newsgroup_train.target[train_split:]

    test_data = newsgroup_test.data
    test_targets = newsgroup_test.target

    nlp = NLPPipeline()
    train_data_indices, valid_data_indices, test_data_indices = nlp(train_data, valid_data, test_data)

    train_ds = NewsGroupDataset(train_data_indices, train_targets, max_sentence_length=max_sentence_length)
    valid_ds = NewsGroupDataset(valid_data_indices, valid_targets, max_sentence_length=max_sentence_length)
    test_ds = NewsGroupDataset(test_data_indices, test_targets, max_sentence_length=max_sentence_length)
    dm = DataModule(train_dataset=train_ds, valid_dataset=valid_ds, test_dataset=test_ds, collate_fn=newsgroup_collate_func)
    return dm

@gin.configurable
def start_experiment(output_dir, max_epochs, seed=None):
    set_seed(seed=seed)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tb_logs'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Running on {}".format(device))

    dm = get_datamodule()
    model = BagOfWords()
    optimizer = OptimConfig().create_optimizer(model)
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")  
    trainer = Trainer(optimizer, output_dir, max_epochs, loss_function, tb_writer=writer, device=device)

    trainer.fit(model, dm)


class NewsGroupDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, target_list, max_sentence_length=300):
        """
        @param data_list: list of newsgroup tokens 
        @param target_list: list of newsgroup targets 

        """
        self.data_list = data_list
        self.target_list = target_list
        self.max_sentence_length = max_sentence_length
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """         
        token_idx = self.data_list[key][:self.max_sentence_length] # truncating
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]


if __name__ == "__main__":

    gin_wrap(start_experiment)