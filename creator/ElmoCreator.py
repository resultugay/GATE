import logging
import sys

sys.path.append("..")
from .Creator import Creator
from dataloader.ElmoDataLoader import ElmoDataset
import torch.nn as nn
import torch
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from .LossFunctions import PairWiseLoss
import tensorflow_hub as hub
import tensorflow as tf


class ElmoCreator(Creator):

    def __init__(self, args, data_training):
        self.args = args
        self.df = data_training
        self.training_data = {}
        self.labels = {}
        self.index_word = {}
        self.word_index = {}
        self.prepare_data({})
        self.elmo = hub.load('elmo_3')
        self.CC = {}

    def prepare_data(self, remove_list):
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
            dataset = ElmoDataset(self.training_data[col], self.labels[col], self.elmo, self.index_word[col])
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
            write_every = self.args.write_every
            epoch = self.args.epoch

            model = ElmoNet(1024, self.args.emb_dim)
            criterion = PairWiseLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001)
            # CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
            torch.backends.cudnn.benchmark = True
            total_loss = 0
            for ep in range(epoch):
                for data, label in dataloader:
                    model.zero_grad()
                    data = torch.cat(data, dim=0)
                    data, label = data.to(device), label.to(device)
                    model = model.to(device)
                    res = model(data)
                    loss = criterion(res, label)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                if ((ep + 1) % write_every) == 0:
                    logging.info('Epoch ' + str(ep) + ' Loss ' + str(
                        total_loss / (write_every * len(dataloader))))
                    total_loss = 0

            vector = {}
            for key, value in dataset.embeddings.items():
                inter_vector = torch.matmul(dataset.embeddings[key].reshape(1, -1), model.x_embed.weight.t())
                inter_vector = inter_vector.squeeze()
                vector[key] = inter_vector

            """
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            vec_stat = vector['status']
            embedding_vectors = []
            for key, value in vector.items():
                res = cos(vec_stat, value)
                print(key, res)
            """
            self.save_vectors(col, vector)
            #self.save_img(col,vector)
            """
            word_vectors = {}
            for key, value in dataset.embeddings.items():
                vector = torch.matmul(dataset.embeddings[key].reshape(1, -1), model.x_embed.weight.t())
                word_vectors[key] = vector

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            vec_stat = word_vectors['status']
            embedding_vectors = []
            for key, value in word_vectors.items():
                res = cos(vec_stat, value)
                print(key, res)
            """
            self.create_currency_constraints(col, vector, self.CC)
        return self.CC


class ElmoNet(nn.Module):
    def __init__(self, elmo_dim, embed_dim):
        super(ElmoNet, self).__init__()
        self.x_embed = nn.Linear(elmo_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs_):
        x = self.x_embed(inputs_[0])
        x1 = self.x_embed(inputs_[1])
        x2 = self.x_embed(inputs_[2])
        one = torch.dot(x, x1)
        two = torch.dot(x, x2)
        return (one, two)
