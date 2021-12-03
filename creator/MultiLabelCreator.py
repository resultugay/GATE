import logging
from .Creator import Creator
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("..")

from dataloader.MultiLabelDataLoader import MultiLabelDataset
from collections import Counter

class MultiLabelCreator(Creator):
    columns = None
    df = None
    max_number_of_neuron = 0


    def __init__(self,args):
        logging.info('Creator is MultiLabel')
        columns = args.columns
        df = pd.read_csv(args.dataset)
        max_number_of_neuron = max(Counter(df['id'].to_list()).values())



    def train(self):
        for col in self.columns:
            dataset = MultiLabelDataset(self.df,col)



    def evaluate(self):
        pass


class MultiLabelModel(nn.Module):
    def __init__(self,x):
        pass
