import gin
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from pytorch_wrapper.models.nlp_models import RNNLM
from pytorch_wrapper.data.processing import NLPPipeline
from pytorch_wrapper.data import DataModule
from pytorch_wrapper.optimizer import OptimConfig
from pytorch_wrapper.trainer import Trainer
from pytorch_wrapper import DATA_DIR
from pytorch_wrapper.utils import load_text_data, set_seed, get_device, gin_wrap
from pytorch_wrapper.callbacks.callbacks import Callback


@gin.configurable
def get_datamodule():
    train_data = load_text_data(os.path.join(DATA_DIR, 'ptb.train.nounk.txt'), split_by='\n')
    valid_data = load_text_data(os.path.join(DATA_DIR, 'ptb.valid.txt'), split_by='\n')
    test_data = load_text_data(os.path.join(DATA_DIR, 'ptb.test.txt'), split_by='\n')

    nlp = NLPPipeline()
    train_data_indices, valid_data_indices, test_data_indices = nlp(train_data, valid_data, test_data)

    train_ds = LMDataset(train_data_indices)
    valid_ds = LMDataset(valid_data_indices)
    test_ds =  LMDataset(test_data_indices)
    dm = DataModule(train_dataset=train_ds, valid_dataset=valid_ds, test_dataset=test_ds)
    return dm, nlp

@gin.configurable
def start_experiment(output_dir, max_epochs, seed=None):
    set_seed(seed=seed)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tb_logs'))
    device = get_device()
    dm, nlp = get_datamodule()

    model = RNNLM(vocab_size=len(nlp.id2token), padding_idx=nlp.padding_idx)
    optimizer = OptimConfig().create_optimizer(model)
    trainer = Trainer(optimizer, output_dir, max_epochs, tb_writer=writer, device=device)

    trainer.fit(model, dm)
    

@gin.configurable
class LMDataset(Dataset):
   
    def __init__(self, tokenized_arr, max_seq_length=300, padding_idx=0, unknown_idx=1):
        self.tokenized_arr = tokenized_arr
        self.max_seq_length = max_seq_length
        self.padding_idx = padding_idx
        self.unknown_idx = unknown_idx
        
    def __len__(self):
        return len(self.tokenized_arr)
        
    def __getitem__(self, key):
        # Truncate to max_seq_length
        x = self.tokenized_arr[key][:self.max_seq_length]

        # Pad
        arr = np.pad(x, (0, self.max_seq_length - len(x)), constant_values=self.padding_idx)
        assert len(arr) == self.max_seq_length
        return {
            "arr": arr,
        }

if __name__ == "__main__":

    gin_wrap(start_experiment)