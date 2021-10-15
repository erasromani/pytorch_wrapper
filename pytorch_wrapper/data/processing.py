import gin
import spacy
import string
import dataclasses

from collections import Counter


@gin.configurable
@dataclasses.dataclass
class Tokenizer:
    name: str = 'en_core_web_sm'
    exclude_punctuation: bool = False
    lowercase: bool = False
    bos_tok: str = '<bos>'
    eos_tok: str = '<eos>'

    def __post_init__(self):
        self.nlp = spacy.load(self.name)
        for name in self.nlp.component_names:
            self.nlp.remove_pipe(name)
        if self.exclude_punctuation:
            self.punctuations = string.punctuation

    def tokenize(self, sent):
        sent = sent.strip()
        tokens = self.nlp(sent)
        if self.exclude_punctuation and self.lowercase:
            sent_tokens = [token.text.lower() for token in tokens if (token.text not in self.punctuations)]
        elif self.exclude_punctuation and not self.lowercase:
            sent_tokens = [token.text for token in tokens if (token.text not in self.punctuations)]
        elif not self.exclude_punctuation and self.lowercase:
            sent_tokens = [token.text.lower() for token in tokens]
        elif not self.exclude_punctuation and not self.lowercase:
            sent_tokens = [token.text for token in tokens]
        return [self.bos_tok] + sent_tokens + [self.eos_tok]

    def __call__(self, dataset):
        token_dataset = []
        all_tokens = []

        for sample in dataset:
            tokens = self.tokenize(sample)
            token_dataset.append(tokens)
            all_tokens += tokens

        return token_dataset, all_tokens


@gin.configurable
@dataclasses.dataclass
class NLPPipeline:
    max_vocab_size: int = 10000
    padding_idx: int = 0
    unknown_idx: int = 1
    padding_tok: str = '<pad>'
    unknown_tok: str = '<unk>'

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
        id2token = [self.padding_tok, self.unknown_tok] + id2token
        token2id[self.padding_tok] = self.padding_idx
        token2id[self.unknown_tok] = self.unknown_idx
        return token2id, id2token
    
    def token2index(self, tokens_data):
        indices_data = []
        for tokens in tokens_data:
            index_list = [self.token2id[token] if token in self.token2id else self.unknown_idx for token in tokens]
            indices_data.append(index_list)
        return indices_data

    def __call__(self, train_data, valid_data, test_data=None):
        train_data_tokens, all_train_tokens = self.tokenizer(train_data)
        valid_data_tokens, _ = self.tokenizer(valid_data)
        self.token2id, self.id2token = self.build_vocab(all_train_tokens)
        train_data_indices = self.token2index(train_data_tokens)
        valid_data_indices = self.token2index(valid_data_tokens)
        if test_data is not None:
            test_data_tokens, _  = self.tokenizer(test_data)
            test_data_indices  = self.token2index(test_data_tokens)
            return train_data_indices, valid_data_indices, test_data_indices
        return train_data_indices, valid_data_indices
