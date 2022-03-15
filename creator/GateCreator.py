import logging
import sys
from creator.GateDataset import GateDataset
import torch.nn as nn
import torch
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time

sys.path.append("..")
from .Creator import Creator


class GateCreator(Creator):

    def __init__(self, args, training_data):
        self.training_data = training_data
        self.args = args

    def train(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        model = Net(1536, 80, 2)
        model = model.to(device)
        criterion = PairWiseLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        training_set = GateDataset(self.training_data)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=self.args.batch_size)
        for ep in range(self.args.epoch):
            total_loss = 0
            start_time = time.time()
            for local_batch in training_generator:
                model.zero_grad()
                local_batch = torch.stack(local_batch)
                local_batch = local_batch.to(device)
                res = model(local_batch)
                loss = criterion(res)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60
            logging.info('Epoch: ' + str(ep) + ' Total Loss: %.4f | execution time %.4f mins' , total_loss, epoch_time)


# Here we define our NN module
class Net(nn.Module):
    def __init__(self, dim, hidden_dim, embed_dim, ):
        super(Net, self).__init__()
        self.x_embed = nn.Linear(dim, hidden_dim)
        self.x_embed2 = nn.Linear(hidden_dim, embed_dim)
        self.embed_size = dim
        self.relu = nn.ReLU()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, inputs_):
        x = self.x_embed(inputs_[0])
        x = self.x_embed2(x)

        x1 = self.x_embed(inputs_[1])
        x1 = self.x_embed2(x1)

        x2 = self.x_embed(inputs_[2])
        x2 = self.x_embed2(x2)

        one = self.cos(x, x1)
        two = self.cos(x, x2)
        return one, two


class PairWiseLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PairWiseLoss, self).__init__()

    def forward(self, res):
        return torch.sum(torch.max(torch.tensor(0), 1 - res[0]) + torch.max(torch.tensor(0), res[1]))
