import gin
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from attrdict import AttrDict

from pytorch_wrapper.models import nnModule

def sample_from_distribution(dist):
    tokens, p = zip(*dist.items())
    return np.random.choice(tokens, p=p)


class Node:
    def __init__(self):
        self.count = 0
        self.child_nodes = {}

    def has_token(self, token):
        return token in self.child_nodes

    def create_child_node(self, token):
        self.child_nodes[token] = Node()

    def increment_count(self):
        self.count += 1
    
    def get_child_node(self, token):
        if token not in self.child_nodes:
            raise KeyError(f"'{token}' not found in child node")
        return self.child_nodes[token]
        

class NgramTrie:
    def __init__(self, n):
        self.n = n
        self.root_node = Node()
        self.vocab = set()
    
    def add_ngram(self, ngram):
        if len(ngram) > self.n:
            raise RuntimeError(f"This NgramTrie only supports up to {n}-grams")

        # Update vocabulary
        self.vocab.update(ngram)

        # Update Trie
        current_node = self.root_node
        current_node.increment_count()
        for token in ngram:
            if not current_node.has_token(token):
                current_node.create_child_node(token)
            current_node = current_node.get_child_node(token)
            current_node.increment_count()
        
    def fit(self, lines):
        for line in lines:
            sequence = ["<bos>"] + line.split() + ["<eos>"]
            for i in range(len(sequence)-self.n+1):
                # for each ngram (sliding window)
                ngram = tuple(sequence[i:i+self.n])
                self.add_ngram(ngram)
    
    def get_ngram_cond_dist(self, input_ngram, variant=None):
        """Get n-gram conditional distribution from trie.

        Arguments:
        input_ngram: Tuple of n-1 tokens

        Returns:
        Dictionary mapping each possible token->conditional probability
        """
        current_node = self.root_node
        if variant == "backoff":
            # Repeatedly back-off to smaller N
            while True:
                current_node = self.root_node
                matched = True
                for token in input_ngram:
                    # Did not find token: reduce n-gram
                    if not current_node.has_token(token):
                        input_ngram = input_ngram[1:]
                        matched = False
                        break
                    current_node = current_node.get_child_node(token)

                # Successfully matched n-gram, break out of loop
                if matched:
                    break

            if len(input_ngram) != self.n:
                print(f"Fell back to {len(input_ngram)+1}-gram")
        elif variant == "smoothing":
                try:
                    # Traverse trie
                    for token in input_ngram:
                        current_node = current_node.get_child_node(token)
                    # Get count for each child node
                    child_counts = {}
                    for token, node in current_node.child_nodes.items():
                        child_counts[token] = node.count
                except KeyError:
                    # Unseen N-gram
                    child_counts = {}
        elif variant is None:
            for token in input_ngram:
                current_node = current_node.get_child_node(token)
        else:
            raise ValueError("invalid variant {} entered".format(variant))

        # Get count for each child node
        if variant == "smoothing":
            for token in self.vocab:
                if token not in child_counts:
                    child_counts[token] = 0
                child_counts[token] += 1
        else:
            child_counts = {}
            for token, node in current_node.child_nodes.items():
                child_counts[token] = node.count

        # Divide by total counts
        total_count = sum(child_counts.values())
        child_prob = {}
        for token, count in child_counts.items():
            child_prob[token] = count / total_count
        return child_prob

    def sample_from_trie(self, seed_tokens, cond_dist_func, max_length=20):
        all_tokens = seed_tokens.copy()
        for i in range(max_length):
            input_tokens = all_tokens[-(self.n-1):]
            dist = cond_dist_func(input_tokens)
            new_token = sample_from_distribution(dist)
            all_tokens.append(new_token)
            if new_token == "<eos>":
                break
        return all_tokens


@gin.configurable
class NgramLM(nnModule):
    def __init__(self, vocab_size, ngram_n, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim * (ngram_n - 1), emb_dim)
        self.out_embed = nn.Linear(emb_dim, vocab_size)

        # Share weights between embeddings
        self.out_embed.weight = self.in_embed.weight
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        out = self.in_embed(x)
        out = out.reshape(batch_size, seq_len * self.emb_dim)
        out = F.relu(self.linear(out))
        out = self.out_embed(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y, reduction='mean')
        return loss
            
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y, reduction='sum')
        pred_indices = pred.max(dim=1).indices
        correct = (pred_indices==y).float().sum()
        self.n_examples += y.size(0)
        return {'loss': loss, 'correct': correct}
        
    def on_validation_start(self, trainer):
        self.logs = []
        self.n_examples = 0

    def on_validation_batch_end(self, trainer, outputs, batch, batch_idx):
        self.logs.append(outputs)

    def on_validation_end(self, trainer):
        loss = torch.stack([log['loss'] for log in self.logs]).sum() / self.n_examples
        accuracy = torch.stack([log['correct'] for log in self.logs]).float().sum() / self.n_examples
        return {'loss': loss, 'accuracy': accuracy}


@gin.configurable
class RNNLM(nnModule):
    def __init__(self, vocab_size, emb_dim, num_layers, dropout_p=0.5, padding_idx=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.out_embed = nn.Linear(emb_dim, vocab_size)
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        # Share weights between embeddings
        self.out_embed.weight = self.in_embed.weight
    
    def forward(self, x):
        out = self.in_embed(x)
        # h_0 is all zeros if it is not provided
        out, hidden = self.rnn(out)
        out = self.out_embed(self.dropout(out))
        return out

    def training_step(self, batch, batch_idx):
        data = batch['arr'][:, :-1]
        target = batch['arr'][:, 1:]
        logits = self(data)
        flattened_logits = logits.reshape(-1, self.vocab_size)
        flattened_labels = target.reshape(-1)
        loss = F.cross_entropy(
            flattened_logits,
            flattened_labels,
            reduction='mean',
            ignore_index=self.padding_idx,
        )
        return loss
            
    def validation_step(self, batch, batch_idx):
        data = batch['arr'][:, :-1]
        target = batch['arr'][:, 1:]
        logits = self(data)
        flattened_logits = logits.reshape(-1, self.vocab_size)
        flattened_labels = target.reshape(-1)
        loss = F.cross_entropy(
            flattened_logits,
            flattened_labels,
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        # Only count non-padding tokens
        valid_indices = flattened_labels != self.padding_idx
        self.n_examples += valid_indices.sum()
        preds = flattened_logits.argmax(-1)
        raw_correct_preds = (preds == flattened_labels)
        correct = (raw_correct_preds & valid_indices).sum()
        return {'loss': loss, 'correct': correct}
        
    def on_validation_start(self, trainer):
        self.logs = []
        self.n_examples = 0

    def on_validation_batch_end(self, trainer, outputs, batch, batch_idx):
        self.logs.append(outputs)

    def on_validation_end(self, trainer):
        loss = torch.stack([log['loss'] for log in self.logs]).sum() / self.n_examples
        accuracy = torch.stack([log['correct'] for log in self.logs]).float().sum() / self.n_examples
        return {'loss': loss, 'accuracy': accuracy}


@gin.configurable
class BagOfWords(nnModule):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim, num_class, padding_idx=0):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        self.embed = nn.Embedding(vocab_size+2, emb_dim, padding_idx=padding_idx)
        self.linear = nn.Linear(emb_dim,num_class)
        self.config = AttrDict({
            'vocab_size': vocab_size,
            'emb_dim': emb_dim,
            'padding_idx': padding_idx,
        })
    
    def forward(self, data, length):
        """
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each document in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()     
        out = self.linear(out.float())
        return out

    def training_step(self, batch, batch_idx):
        data, length, target = batch
        output = self(data, length)
        loss = torch.nn.CrossEntropyLoss(reduction="mean")(output, target)
        return loss
            
    def validation_step(self, batch, batch_idx):
        data, length, target = batch
        output = self(data, length)
        loss = torch.nn.CrossEntropyLoss(reduction="sum")(output, target)
        output = F.softmax(output, dim=1)
        predicted = output.max(1, keepdim=True)[1]                
        self.n_examples += target.size(0)
        correct = predicted.eq(target.view_as(predicted)).sum()
        return {'loss': loss, 'correct': correct}
        
    def on_validation_start(self, trainer):
        self.logs = []
        self.n_examples = 0

    def on_validation_batch_end(self, trainer, outputs, batch, batch_idx):
        self.logs.append(outputs)

    def on_validation_end(self, trainer):
        loss = torch.stack([log['loss'] for log in self.logs]).sum() / self.n_examples
        accuracy = torch.stack([log['correct'] for log in self.logs]).float().sum() / self.n_examples
        return {'loss': loss, 'accuracy': accuracy}
