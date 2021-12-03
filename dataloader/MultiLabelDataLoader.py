import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import logging

class MultiLabelDataset(torch.utils.data.Dataset):

    def __init__(self,df,column):
        df.index = df.id
        X = []
        for index in df.index:
            attrs = df.loc[index]['status'].to_list()
            X.append(attrs)

    def __len__(self):
        pass

    def __getitem__(self,idx):
        pass
