# pylint: disable=no-member

from typing import List
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from .utils import get_data_path

class NodeEmbedding:
    def __init__(self, n2v_path):
        self.model = KeyedVectors.load(n2v_path)
        self.vocab = self.model.wv.vocab
        self.PAD_INDEX = len(self.vocab)
        self.UNK_INDEX = len(self.vocab)+1

    @property
    def dim(self):
        return self.model.wv.vector_size

    def get_n2v_embedding_layer(self):
        torch.manual_seed(15532)
        n2v = torch.FloatTensor(self.model.wv.vectors)        
        pad = torch.rand([1, n2v.shape[-1]])
        unk = torch.rand([1, n2v.shape[-1]])
        weights = torch.cat([n2v, pad, unk], axis=0)
        emb = nn.Embedding.from_pretrained(weights)
        return emb

    def stoi(self, s):
        if s == "<PAD>":
            return self.PAD_INDEX
        if s in self.vocab:
            return self.vocab[s].index        
        else:
            return self.UNK_INDEX
    
    def encode(self, tokens: List[str], max_length, drop_unk=True):        
        ids = [self.stoi(x) for x in tokens]
        if drop_unk:
            ids = [x for x in ids if x != self.UNK_INDEX]

        if len(ids) > max_length:
            ids = ids[:max_length]
        pad_length = max_length - len(ids)
        padded = ids + [self.PAD_INDEX] * pad_length
        return padded
