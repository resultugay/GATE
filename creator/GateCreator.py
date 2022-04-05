import logging
import sys
from creator.GateDataset import GateDataset, GateValidationTestDataset
import torch.nn as nn
import torch
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import sys
import numpy as np
from sklearn.metrics import ndcg_score

sys.path.append("..")
from .Creator import Creator


class GateCreator(Creator):

    def __init__(self, args, training_data):
        self.args = args
        self.model = Net(self.args.input_dim, self.args.embedding_dim)

    def collate_fn(self, validation_set):
        max_len = max([len(x) for x in validation_set])
        lengths = [len(x) for x in validation_set]
        lengths = torch.tensor(lengths)

        zeros = torch.zeros(validation_set[0][0].shape)
        data = []
        for v in validation_set:
            temp = v.copy()
            for i in range(len(v), max_len):
                temp.append(zeros)
            data.append(temp)
        return data, lengths

    def train(self, training_data, validation_data):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        model = self.model.to(device)
        criterion = PairWiseLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        training_set = GateDataset(training_data)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=self.args.batch_size)
        total_loss_min = sys.maxsize

        validation_set = GateValidationTestDataset(validation_data)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=self.args.batch_size, collate_fn=self.collate_fn)

        for ep in range(self.args.epoch):
            total_loss = 0
            ndcg = 0
            start_time = time.time()
            model.train()
            for local_batch in training_generator:
                model.zero_grad()
                local_batch = torch.stack(local_batch)
                local_batch = local_batch.to(device)
                res = model(local_batch)
                loss = criterion(res)
                total_loss += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60
            logging.info('Epoch: ' + str(ep) + ' Total Loss: %.4f | execution time %.4f mins', total_loss, epoch_time)

            model.eval()
            for local_batch, lengths in validation_generator:
                for attributes, length in zip(local_batch, lengths):
                    attributes[0] = attributes[0].to(device)
                    attr_emb = model.x_embed2last(model.x_input2embed(attributes[0]))
                    res = []
                    for idx in range(1, length):
                        attributes[idx] = attributes[idx].to(device)
                        val_emb = model.x_embed2last(model.x_input2embed(attributes[idx]))
                        dist = np.linalg.norm(attr_emb.detach().numpy() - val_emb.detach().numpy())
                        res.append(dist)

                    pred = np.argsort(res) + 1
                    ground_truth = np.arange(length - 1) + 1
                    ndcg += ndcg_score([ground_truth], [pred])

            logging.info('NDCG score for validation set is ' + str(ndcg / (len(validation_generator)*self.args.batch_size)))

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
