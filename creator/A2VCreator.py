import logging
import sys

sys.path.append("..")
from .Creator import Creator
from dataloader.A2VDataLoader import AV2Dataset
import torch.nn as nn
import torch
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from .LossFunctions import A2VLoss

class A2VCreator(Creator):
    def __init__(self, args,data_training):
        self.args = args
        self.df = data_training
        self.training_data = {}
        self.labels = {}
        self.index_word = {}
        self.word_index = {}
        self.prepare_data()

    def prepare_data(self,remove_list):
        self.df.index = self.df.id
        for col in self.args.columns:
            logging.info('Preparing training data for column ' + col)
            self.training_data[col] = []
            self.labels[col] = []
            tokens = set(self.df[col].unique())
            tokens.add(col)
            index_word = {i: x for i, x in enumerate(tokens)}
            word_index = {x: i for i, x in enumerate(tokens)}
            self.index_word[col] = index_word
            self.word_index[col] = word_index

            for index in self.df.index:
                sub_df = self.df.loc[index][[col, 'timestamp']]
                status = dict(zip(sub_df[col], sub_df.timestamp))

                latest_timestamp = sub_df['timestamp'].max()

                key_attr = word_index[col]
                for key, value in status.items():
                    for key2, value2 in status.items():
                        if (value == latest_timestamp and value2 != latest_timestamp) and (key != key2) and key_attr != \
                                word_index[key] and key_attr != word_index[key2]:
                            if remove_list.get(col,None) and (key, key2) in remove_list[col]:
                                #logging.info(str(key) + ' and ' + str(key2) + ' removed from training data')
                                #Consider these as negative instances
                                self.training_data[col].append(
                                    torch.tensor([word_index[col], word_index[key2], word_index[key]]))
                                self.labels[col].append(int(status[key2]) - int(status[key]))
                            else:
                                self.training_data[col].append(
                                    torch.tensor([word_index[col], word_index[key], word_index[key2]]))
                                self.labels[col].append(int(status[key]) - int(status[key2]))
                                #logging.info(str(key) + ' and ' + str(key2) + ' removed from training data')


    def train(self):
        for col in self.args.columns:
            logging.info('Current Column is ' + col)
            dataset = AV2Dataset(self.training_data[col], self.labels[col])
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

            emb_dim = self.args.emb_dim
            vocab_size = len(self.word_index[col])
            write_every = self.args.write_every
            epoch = self.args.epoch

            model = A2VNet(vocab_size, emb_dim)
            criterion = A2VLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001)
            # CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
            torch.backends.cudnn.benchmark = True
            total_loss = 0
            for ep in range(epoch):
                for data, label in dataloader:
                    data, label = data.to(device), label.to(device)
                    model = model.to(device)
                    model.zero_grad()
                    data = data.squeeze()
                    res = model(data)
                    loss = criterion(res, label)
                    loss.backward()
                    total_loss += loss.item()
                    optimizer.step()
                if ((ep + 1) % write_every) == 0:
                    logging.info('Epoch ' + str(ep) + ' Loss ' + str(total_loss / (
                                write_every * len(dataloader))))
                    total_loss = 0

            vector = {}
            for key, value in self.index_word[col].items():
                vector[value] = model.embeddings.weight[key]

            self.save_img(model,col,vector)
            self.save_vectors(col,vector)

class A2VNet(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(A2VNet, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim, 1)
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, inputs_):
        x = self.embeddings(inputs_[0])
        x1 = self.embeddings(inputs_[1])
        x2 = self.embeddings(inputs_[2])
        one = self.cos(x, x1)
        two = self.cos(x, x2)
        return two - one



