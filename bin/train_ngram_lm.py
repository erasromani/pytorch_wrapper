import os
import gin
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from pytorch_wrapper import DATA_DIR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from pytorch_wrapper.utils import get_device, gin_wrap, set_seed
from pytorch_wrapper.data import DataModule
from pytorch_wrapper.models.nlp_models import NgramLM
from pytorch_wrapper.trainer import Trainer
from pytorch_wrapper.optimizer import OptimConfig

logger = logging.getLogger(__name__)

def load_tokens(path):
    with open(path, "r") as f:
        tokens_list = []
        for line in f:
            line = line.strip()
            if not line: 
                continue
            tokens_list.append(["<bos>"] + line.split() + ["<eos>"])
    return tokens_list

def get_ngrams_arr(tokens_list, tok2idx, n, unknown_tok="<unk>"):
    ngram_rows = []
    for tokens in tokens_list:
        for i in range(len(tokens)-n+1):
            ngram = tokens[i:i+n]
            ngram_indices = [tok2idx.get(token, tok2idx[unknown_tok]) for token in ngram]
            ngram_rows.append(ngram_indices)
    return np.array(ngram_rows)

@gin.configurable
def get_datamodule(ngram_n, unknown_tok="<unk>"):
    train_tokens = load_tokens(os.path.join(DATA_DIR, "ptb.train.nounk.txt"))
    val_tokens = load_tokens(os.path.join(DATA_DIR, "ptb.valid.txt"))
    test_tokens = load_tokens(os.path.join(DATA_DIR, "ptb.valid.txt"))
    vocab = sorted(list({token for token_line in train_tokens for token in token_line})) + [unknown_tok]
    tok2idx = {vocab: i for i, vocab in enumerate(vocab)}
    idx2tok = dict(enumerate(vocab))
    train_arr = get_ngrams_arr(train_tokens, tok2idx, n=ngram_n)
    val_arr = get_ngrams_arr(val_tokens, tok2idx, n=ngram_n)
    test_arr = get_ngrams_arr(test_tokens, tok2idx, n=ngram_n)

    train_dataset = NgramDataset(train_arr)
    val_dataset = NgramDataset(val_arr)
    test_dataset = NgramDataset(test_arr)

    dm = DataModule(train_dataset=train_dataset, valid_dataset=val_dataset, test_dataset=test_dataset)
    return dm, vocab


class NgramDataset(Dataset):
   
    def __init__(self, tokenized_arr):
        self.tokenized_arr = tokenized_arr

    def __len__(self):
        return self.tokenized_arr.shape[0]
        
    def __getitem__(self, key):
        return self.tokenized_arr[key, :-1], self.tokenized_arr[key, -1]


@gin.configurable
def start_experiment(output_dir, max_epochs, seed=None):
    set_seed(seed=seed)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tb_logs'))
    device = get_device()
    dm, vocab = get_datamodule()

    model = NgramLM(vocab_size=len(vocab))
    optimizer = OptimConfig().create_optimizer(model)
    trainer = Trainer(optimizer, output_dir, max_epochs, tb_writer=writer, device=device)

    trainer.fit(model, dm)


if __name__ == "__main__":

    gin_wrap(start_experiment)

