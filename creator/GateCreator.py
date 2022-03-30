import logging
import sys
from creator.GateDataset import GateDataset
import torch.nn as nn
import torch
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import sys

sys.path.append("..")
from .Creator import Creator


class GateCreator(Creator):

    def __init__(self, args, training_data):
        self.args = args
        self.model = Net(self.args.input_dim, self.args.embedding_dim)

    def train(self, training_data):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        model = self.model.to(device)
        criterion = PairWiseLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        training_set = GateDataset(training_data)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=self.args.batch_size)
        total_loss_min = sys.maxsize
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
            logging.info('Epoch: ' + str(ep) + ' Total Loss: %.4f | execution time %.4f mins', total_loss, epoch_time)
            if total_loss < total_loss_min:
                logging.info('Saving the best model')
                torch.save(model.state_dict(), 'best_saved_weights.pt')
                logging.info('Model saved as best_saved_weights.pt')
                total_loss_min = total_loss
        self.model = model


# Here we define our NN module
class Net(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Net, self).__init__()
        self.x_input2embed = nn.Linear(input_dim, embed_dim)
        self.x_embed2last = nn.Linear(embed_dim, 2)
        self.relu = nn.ReLU()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, inputs_):
        dummy_reference_vector = self.x_input2embed(inputs_[0])
        dummy_reference_vector = self.x_embed2last(dummy_reference_vector)

        latest_value_vector = self.x_input2embed(inputs_[1])
        latest_value_vector = self.x_embed2last(latest_value_vector)

        non_latest_value_vector = self.x_input2embed(inputs_[2])
        non_latest_value_vector = self.x_embed2last(non_latest_value_vector)

        one = self.cos(dummy_reference_vector, latest_value_vector)
        two = self.cos(dummy_reference_vector, non_latest_value_vector)
        return one, two


class PairWiseLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PairWiseLoss, self).__init__()

    def forward(self, res):
        return torch.sum(torch.max(torch.tensor(0), 1 - res[0]) + torch.max(torch.tensor(0), 1 + res[1]))
