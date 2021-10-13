import os
import datetime
import spacy
import string
import dataclasses
import gin
import argh
import logging

from collections import Counter

logger = logging.getLogger(__name__)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError(f'{s} is not a valid boolean string')
    return s == 'True'

def get_output_dir(parent_dir):
    today = datetime.date.today()
    date = today.strftime('%Y-%m-%d')
    now = datetime.datetime.now()
    time = now.strftime('%H-%M-%S')
    output_dir = os.path.join(parent_dir, date, time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Creating folder {}".format(output_dir))
    return output_dir

def split_dataset(split):
    pass

def show_sample():
    pass

def gin_wrap(fnc):
    def main(save_path, config, bindings=""):
        # You can pass many configs (think of them as mixins), and many bindings. Both ";" separated.
        gin.parse_config_files_and_bindings(config.split("#"), bindings.replace("#", "\n"))

    argh.dispatch_command(main)


@dataclasses.dataclass
@gin.configurable
class Tokenizer:
    name: str = 'en_core_web_sm'
    exclude_punctuation: bool = False
    lowercase: bool = False

    def __post_init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.pipeline=[]
        if self.exclude_punctuation:
            self.punctuations = string.punctuation

    def tokenize(self, sent):
        tokens = self.nlp(sent)
        if self.exclude_punctuation and self.lowercase:
            return [token.text.lower() for token in tokens if (token.text not in self.punctuations)]
        elif self.exclude_punctuation and not self.lowercase:
            return [token.text for token in tokens if (token.text not in self.punctuations)]
        elif not self.exclude_punctuation and self.lowercase:
            return [token.text.lower() for token in tokens]
        elif not self.exclude_punctuation and not self.lowercase:
            return [token.text for token in tokens]

    def __call__(self, dataset):
        token_dataset = []
        all_tokens = []
        
        for sample in dataset:
            tokens = self._tokenize(sample)
            token_dataset.append(tokens)
            all_tokens += tokens

        return token_dataset, all_tokens


@dataclasses.dataclass
@gin.configurable
class NLPPipeline:
    max_vocab_size: int = 10000
    padding_idx: int = 0
    unknown_idx: int = 1

    def __post_init__(self):
        self.tokenizer = Tokenizer()
    
    def build_vocab(self, all_tokens):
        # Returns:
        # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
        # token2id: dictionary where keys represent tokens and corresponding values represent indices
        token_counter = Counter(all_tokens)
        vocab, count = zip(*token_counter.most_common(self.max_vocab_size))
        id2token = list(vocab)
        token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
        id2token = ['<pad>', '<unk>'] + id2token
        token2id['<pad>'] = self.padding_idx
        token2id['<unk>'] = self.unknown_idx
        return token2id, id2token
    
    def __call__(self, train_data, valid_data, test_data):
        train_data_tokens, all_train_tokens = self.tokenizer(train_data)
        valid_data_tokens, _ = self.tokenizer(valid_data)
        test_data_tokens, _  = self.tokenizer(test_data)
        self.token2id, self.id2token = self.build_vocab(all_train_tokens)
        train_data_indices = self.token2index(train_data_tokens)
        valid_data_indices = self.token2index(valid_data_tokens)
        test_data_indices  = self.token2index(test_data_tokens)
        return train_data_indices, valid_data_indices, test_data_indices

    def token2index(self, tokens_data):
        indices_data = []
        for tokens in tokens_data:
            index_list = [self.token2id[token] if token in self.token2id else self.unknown_idx for token in tokens]
            indices_data.append(index_list)
        return indices_data


