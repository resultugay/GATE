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
import numpy as np
import matplotlib.pyplot as plt


class A2VCreator(Creator):
    def __init__(self, args,data_training):
        self.args = args
        self.df = data_training
        self.training_data = {}
        self.labels = {}
        self.index_word = {}
        self.word_index = {}
        self.prepare_data()

    def prepare_data(self):
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
                            self.training_data[col].append(torch.tensor([word_index[col], word_index[key], word_index[key2]]))
                            self.labels[col].append(status[key] - status[key2])

    def save_img(self,model,col,vector):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        vec_stat = vector[col]
        embedding_vectors = []
        for key, value in vector.items():
            res = cos(vec_stat, value)
            embedding_vectors.append(value.detach().numpy())

        embedding_vectors = np.array(embedding_vectors)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(embedding_vectors[1:, 0], embedding_vectors[1:, 1], c='white')

        for idx, word in sorted(self.index_word[col].items()):
            x_coord = embedding_vectors[idx, 0]
            y_coord = embedding_vectors[idx, 1]
            ax.annotate(
                word,
                (x_coord, y_coord),
                horizontalalignment='center',
                verticalalignment='center',
                size=20,
                alpha=0.7
            )
            ax.set_title(f"Column-{col}")
        plt.savefig(f"Column-{col}.jpg")

    def save_vectors(self,col,vector):
        f = open('output_vectors/'+ col + "_vectors.txt", "w")
        for key, value in vector.items():
            f.write(key + ' ')
            for elem in value:
                f.write("%s " % elem.item())
            f.write('\n')
        f.close()

    def train(self):
        for col in self.args.columns:
            logging.info('Current Column is ' + col)
            dataset = AV2Dataset(self.training_data[col], self.labels[col])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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
                    logging.info('Epoch ' + str(ep) + ' Loss ' + str(total_loss / (write_every)*len(self.training_data[col])))
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


class A2VLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(A2VLoss, self).__init__()

    def forward(self, input_, target_, smooth=1):
        return torch.exp(input_)
