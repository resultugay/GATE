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


class A2VCreator(Creator):
    def __init__(self, args):
        self.epoch = args.epoch
        status = {
            'dead': 1,
            'deceased': 1,
            'working': 0,
            'operating': 0,
            'functioning': 0,
            'passed away': 1,
            'occupied': 0,
            'running': 0,
            'laboring': 0,
            'status': 0
        }
        tokens = set([s for s in status.keys()])
        self.index_word = {i: x for i, x in enumerate(tokens)}
        self.word_index = {x: i for i, x in enumerate(tokens)}
        self.training_data = []
        key_attr = self.word_index['status']
        for key, value in status.items():
            for key2, value2 in status.items():
                if (value == 1 and value2 != 1) and (key != key2) and key_attr != self.word_index[key] and key_attr != \
                        self.word_index[key2]:
                    self.training_data.append(
                        torch.tensor([self.word_index['status'], self.word_index[key], self.word_index[key2]]))

    def train(self):
        labels = [1] * (len(self.training_data) + 1)
        dataset = AV2Dataset(self.training_data, labels)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        EMD_DIM = 2
        VOCAB_SIZE = len(self.word_index)
        write_every = 10

        model = A2VNet(VOCAB_SIZE, EMD_DIM)
        criterion = A2VLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        total_loss = 0
        for epoch in range(1, self.epoch):
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
            if (epoch % write_every) == 0:
                # print(epoch,total_loss/write_every)
                logging.info('Epoch ' + str(epoch) + ' Loss ' + str(total_loss / write_every))
                total_loss = 0

        vector = {}
        for key, value in self.index_word.items():
            vector[value] = model.embeddings.weight[key]

        """ #Calculate embeddings cosine similarities
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        vec_stat = vector['status']
        embedding_vectors = []
        for key, value in vector.items():
            res = cos(vec_stat, value)
            embedding_vectors.append(value.detach().numpy())
        embedding_vectors = np.array(embedding_vectors)
        """
        f = open('status' + "_vectors.txt", "w")
        for key, value in vector.items():
            f.write(key + ' ')
            for elem in value:
                f.write("%s " % elem.item())
            f.write('\n')
        f.close()


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
