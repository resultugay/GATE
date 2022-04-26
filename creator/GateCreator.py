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
from torchmetrics.functional import retrieval_reciprocal_rank, retrieval_normalized_dcg
sys.path.append("..")
from .Creator import Creator


class GateCreator(Creator):

    def __init__(self, args):
        self.args = args
        self.model = Net(self.args.input_dim, self.args.embedding_dim)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        self.model = self.model.to(self.device)

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

    def train(self, improved_data, training_data, validation_data):
        criterion = PairWiseLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        training_set = GateDataset(improved_data,training_data)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=self.args.batch_size)
        total_loss_min = sys.maxsize

        validation_set = GateValidationTestDataset(validation_data)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=self.args.batch_size,
                                                           collate_fn=self.collate_fn)

        for ep in range(self.args.epoch):
            total_loss = 0
            ndcg = 0
            ndcgmetric = 0
            mrr = 0
            start_time = time.time()
            self.model.train()
            for local_batch in training_generator:
                self.model.zero_grad()
                local_batch = torch.stack(local_batch)
                local_batch = local_batch.to(self.device)
                res = self.model(local_batch)
                loss = criterion(res)
                total_loss += loss.item() / self.args.batch_size
                loss.backward()
                optimizer.step()
            end_time = time.time()
            epoch_time = (end_time - start_time) / 60
            logging.info('Epoch: ' + str(ep) + ' Total Loss: %.4f | execution time %.4f mins', total_loss, epoch_time)

            self.model.eval()
            for local_batch, lengths in validation_generator:
                for attributes, length in zip(local_batch, lengths):
                    attributes[0] = attributes[0].to(self.device)
                    attr_emb = self.model.x_embed2last(self.model.x_input2embed(attributes[0]))
                    res = []
                    for idx in range(1, length):
                        attributes[idx] = attributes[idx].to(self.device)
                        val_emb = self.model.x_embed2last(self.model.x_input2embed(attributes[idx]))
                        dist = np.linalg.norm(attr_emb.detach().numpy() - val_emb.detach().numpy())
                        res.append(dist)

                    pred = np.argsort(res) + 1
                    ground_truth = np.arange(length - 1) + 1
                    ndcg += ndcg_score([ground_truth], [pred])
                    ndcgmetric += retrieval_normalized_dcg(torch.tensor(pred,dtype=torch.float32), torch.tensor(ground_truth))

                    mrr_pred = [0]*len(pred)
                    mrr_pred[np.argmin(pred)] = 1
                    mrr_ground_truth = [True] + [False]*(len(pred)-1)
                    mrr_pred = torch.tensor(mrr_pred,dtype=torch.float32)
                    mrr_ground_truth = torch.tensor(mrr_ground_truth)
                    mrr += retrieval_reciprocal_rank(mrr_pred,mrr_ground_truth)

            logging.info(
                'NDCG score for validation set is ' + str(ndcg / (len(validation_generator) * self.args.batch_size)))

            logging.info(
                'NDCG score from torchmetrics for validation set is ' + str(ndcgmetric.item() / (len(validation_generator) * self.args.batch_size)))

            logging.info(
                'MRR score for validation set is ' + str(mrr.item() / (len(validation_generator) * self.args.batch_size)))

            if total_loss < total_loss_min:
                logging.info('Saving the best model')
                torch.save(self.model.state_dict(), 'best_saved_weights.pt')
                logging.info('Model saved as best_saved_weights.pt')
                total_loss_min = total_loss


# Here we define our NN module
class Net(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Net, self).__init__()
        self.x_input2embed = nn.Linear(input_dim, embed_dim)
        self.x_embed2last = nn.Linear(embed_dim, 2)
        self.relu = nn.ReLU()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs_):
        dummy_reference_vector = self.x_input2embed(inputs_[0].float())
        dummy_reference_vector = self.relu(self.x_embed2last(dummy_reference_vector))
        dummy_reference_vector = self.dropout(dummy_reference_vector)

        latest_value_vector = self.x_input2embed(inputs_[1].float())
        latest_value_vector = self.relu(self.x_embed2last(latest_value_vector))
        latest_value_vector = self.dropout(latest_value_vector)

        non_latest_value_vector = self.x_input2embed(inputs_[2].float())
        non_latest_value_vector = self.relu(self.x_embed2last(non_latest_value_vector))
        non_latest_value_vector = self.dropout(non_latest_value_vector)

        dummy_reference_vector = dummy_reference_vector.double()
        latest_value_vector = latest_value_vector.double()
        non_latest_value_vector = non_latest_value_vector.double()
        # https://discuss.pytorch.org/t/dot-product-batch-wise/9746

        # Adaptive Margin Dot Product
        one = torch.sum(dummy_reference_vector*latest_value_vector,dim=1)
        two = torch.sum(dummy_reference_vector*non_latest_value_vector,dim=1)

        cos_one = self.cos(dummy_reference_vector, latest_value_vector)
        cos_two = self.cos(dummy_reference_vector, non_latest_value_vector)

        return one, cos_one, two, cos_two


class PairWiseLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PairWiseLoss, self).__init__()

    def forward(self, res):
        #return torch.sum( torch.max(torch.tensor(0), (1 - res[0] - res[1])) + torch.max(torch.tensor(0),(1 + res[2] - res[3])))
        return torch.sum( torch.max(torch.tensor(0), (res[1])) + torch.max(torch.tensor(0),(1 - res[3])))
