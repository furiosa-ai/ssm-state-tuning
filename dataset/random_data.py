import transformers
from transformers import TapexTokenizer
from transformers.models.auto import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

from dataset.collator import DataCollator
from .base import DatasetBase
import evaluate
import numpy as np
import pandas as pd
import torch


class RandomDataset(DatasetBase):
    def __init__(self, tokenizer: AutoTokenizer, seqlen, label_seqlen_ratio=0.1, size=None, split="train", use_cache=True, **kwargs):
        path = None
        self.seqlen = seqlen
        self.label_seqlen = int(seqlen * label_seqlen_ratio)
        self.input_seqlen = self.seqlen - self.label_seqlen
        self.size = size if size is not None else 2000

        super().__init__(tokenizer, path, split,  
                         use_cache=False, **kwargs)
    def __len__(self):
        return self.size

    def get_ids(self, idx):
        return [torch.randint(0, 1000, [l]) for l in (self.input_seqlen, self.label_seqlen)]
    
    def compute_metrics(self, eval_preds):
        return {}
    
    def get_input_label(self, idx):
        pass

    def preproc_input_label(self, input, label):
        pass


class RandomDataModule:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.dataset = RandomDataset(tokenizer=tokenizer, **kwargs)
        self.data_collator = DataCollator(tokenizer=tokenizer)
