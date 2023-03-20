import numpy as np
import pandas as pd
import torch
import transformers as ppb  # pytorch transformers
import pickle

# Load pretrained model/tokenizer
model_class, tokenizer_class, pretrained_weights = (
ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
# if we want to use BERT instead of DistilBert, use the following line
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


def get_emb(word):
    tokenized_text = tokenizer.encode(word, add_special_tokens=True)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([tokenized_text])
    segments_tensors = torch.tensor([segments_ids])
    out = model(tokens_tensor, segments_tensors)
    return out['last_hidden_state'][0][0]
