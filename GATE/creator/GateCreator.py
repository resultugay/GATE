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
from sklearn.metrics import accuracy_score
from sklearn.metrics import ndcg_score
from torchmetrics.functional import retrieval_reciprocal_rank, retrieval_normalized_dcg

sys.path.append("..")
from .Creator import Creator
from metrics import metrics

class GateCreator(Creator):

    def __init__(self, args):
        self.args = args
        self.model = Net(self.args.input_dim, self.args.embedding_dim)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        self.model = self.model.to(self.device)

        # only use for store validationT to early terminate
        self.all_validT_attrs, self.all_validT_vals = [], []

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

    def evaluate(self, improved_data, training_data):
        '''
        Only evaluate the training instances, such that
        :param improved_data:
        :param training_data:
        :return:
        '''
        training_set = GateDataset(improved_data, training_data)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=self.args.batch_size)
        self.model.eval()
        mispredicted_embedds = []
        '''
        for batch in training_generator:
            for oneTOEmbedds in batch:
                if not self.predictOne(oneTOEmbedds):
                    mispredicted_embedds.append(oneTOEmbedds)
        '''
        for oneTOEmbedds in training_set:
            #print('Data : ', oneTOEmbedds)
            if not self.predictOne(oneTOEmbedds):
                mispredicted_embedds.append(oneTOEmbedds)
        return mispredicted_embedds


    def evaluate(self, validationT):
        embedds = validationT['training_embedded']
        all_attrs = [e[0] for e in embedds]
        all_vals_pos = [e[1] for e in embedds]
        all_vals_neg = [e[2] for e in embedds]
        all_attrs = torch.stack(all_attrs)
        all_vals_pos = torch.stack(all_vals_pos)
        all_vals_neg = torch.stack(all_vals_neg)
        predictions = self.predictBatch([all_attrs, all_vals_pos, all_vals_neg])
        labels = [True for i in range(len(predictions))]
        acc = accuracy_score(labels, predictions)
        return acc


    def check(self, to, validation, dataProcessedMap, option, delimitor):
        sc = dataProcessedMap[to.attribute + delimitor + str(to.t0)]
        o1 = validation['data_processed'][sc[0]][sc[1]]
        sc = dataProcessedMap[to.attribute + delimitor + str(to.t1)]
        o2 = validation['data_processed'][sc[0]][sc[1]]
        # get embeddings
        attribute_emb = validation['data_attribute_embeddings'][to.attribute]
        context_index_emb_1 = validation['data_sentence_embeddings'][o1[0]]
        attr_val_emb_1 = validation['data_attribute_embeddings'][str(o1[1])]
        context_index_emb_2 = validation['data_sentence_embeddings'][o2[0]]
        attr_val_emb_2 = validation['data_attribute_embeddings'][str(o2[1])]
        if option == 'creatornc':
            context_index_emb_1 = attr_val_emb_1
            context_index_emb_2 = attr_val_emb_2
        # concate all of the tensors
        attribute_emb = torch.cat((attribute_emb, attribute_emb), 0)
        instance_1 = torch.cat((context_index_emb_1, attr_val_emb_1), 0)
        instance_2 = torch.cat((context_index_emb_2, attr_val_emb_2), 0)
        oneTOEmbedds = [attribute_emb, instance_2, instance_1]
        if self.predictOne(oneTOEmbedds):
            # instance_1 is the latest
            return oneTOEmbedds, [to.attribute, o2, o1, to.eid], to
        else:
            oneTOEmbedds_ = [attribute_emb, instance_1, instance_2]
            to_ = TemporalOrder(to.t1, to.t0, to.attribute, Operator.LESS_THAN, to.eid, 1.0)
            return oneTOEmbedds_, [to.attribute, o1, o2, to.eid], to_


    def predictBatch(self, TOEmbedds):
        attrs = torch.tensor(TOEmbedds[0]).to(self.device)
        attrs_emb = self.model.getAttrEmb(attrs.float())
        vals_pos = torch.tensor(TOEmbedds[1]).to(self.device)
        vals_pos_emb = self.model.getValueEmb(vals_pos.float())
        vals_neg = torch.tensor(TOEmbedds[2]).to(self.device)
        vals_neg_emb = self.model.getValueEmb(vals_neg.float())
        dists_pos = np.linalg.norm(attrs_emb.detach().cpu().numpy() - vals_pos_emb.detach().cpu().numpy(), axis=1)
        dists_neg = np.linalg.norm(attrs_emb.detach().cpu().numpy() - vals_neg_emb.detach().cpu().numpy(), axis=1)
        predicts = []
        for p, n in zip(dists_pos, dists_neg):
            if p < n:
                predicts.append(True)
            else:
                predicts.append(False)
        return predicts

    def predictOne(self, oneTOEmbedds):
        attr = torch.tensor(oneTOEmbedds[0]).to(self.device) #oneTOEmbedds[1], oneTOEmbedds[2]
        attr_emb = self.model.getAttrEmb(attr) #self.model.x_embed2last(self.model.x_input2embed(attr))
        val_pos = torch.tensor(oneTOEmbedds[1]).to(self.device)
        val_pos_emb = self.model.getValueEmb(val_pos) # self.model.x_embed2last(self.model.x_input2embed(val_pos))
        val_neg = torch.tensor(oneTOEmbedds[2]).to(self.device)
        val_neg_emb = self.model.getValueEmb(val_neg) #self.model.x_embed2last(self.model.x_input2embed(val_neg))
        dist_pos = np.linalg.norm(attr_emb.detach().numpy() - val_pos_emb.detach().numpy())
        dist_neg = np.linalg.norm(attr_emb.detach().numpy() - val_neg_emb.detach().numpy())
        if dist_pos < dist_neg:
            return True
        return False

    def predictOneValue(self, attribute_emb, value_emb):
        attr = torch.tensor(attribute_emb).to(self.device)
        attr_emb = self.model.getAttrEmb(attr)
        val = torch.tensor(value_emb).to(self.device)
        val_emb = self.model.getValueEmb(val)
        # print('One value : ', attr_emb.shape, val_emb.shape)
        dist_ = np.linalg.norm(attr_emb.detach().cpu().numpy() - val_emb.detach().cpu().numpy())
        return dist_


    ''' "attribute_embs" and "value_embs" are in the same shape[0], i.e., the same number of records
    '''
    def predictBatchValue(self, attribute_embs, value_embs):
        attrs = torch.tensor(attribute_embs).to(self.device)
        attrs_emb = self.model.getAttrEmb(attrs)
        vals = torch.tensor(value_embs).to(self.device)
        vals_emb = self.model.getValueEmb(vals)
        dist_ = np.linalg.norm(attrs_emb.detach().cpu().numpy() - vals_emb.detach().cpu().numpy(), axis=1)
        return dist_, attrs_emb, vals_emb
        '''
        dist_ = []
        A = attrs_emb.detach().cpu().numpy() - vals_emb.detach().cpu().numpy()
        for a in A:
            dist_.append(np.linalg.norm(a))
        #for a, b in zip(attrs_emb.detach().cpu().numpy(), vals_emb.detach().cpu().numpy()):
            #dist_.append(np.linalg.norm(a - b))
        return dist_, attrs_emb.detach().cpu().numpy(), vals_emb.detach().cpu().numpy()
        '''

    def predictHighConf(self, oneTOEmbedds, confThreshold):
        attr = torch.tensor(oneTOEmbedds[0]).to(self.device)  # oneTOEmbedds[1], oneTOEmbedds[2]
        attr_emb = self.model.getAttrEmb(attr) # self.model.x_embed2last(self.model.x_input2embed(attr))
        val_pos = torch.tensor(oneTOEmbedds[1]).to(self.device)
        val_pos_emb = self.model.getValueEmb(val_pos) # self.model.x_embed2last(self.model.x_input2embed(val_pos))
        val_neg = torch.tensor(oneTOEmbedds[2]).to(self.device)
        val_neg_emb = self.model.getValueEmb(val_neg) # self.model.x_embed2last(self.model.x_input2embed(val_neg))
        dist_pos = np.linalg.norm(attr_emb.detach().cpu().numpy() - val_pos_emb.detach().cpu().numpy())
        dist_neg = np.linalg.norm(attr_emb.detach().cpu().numpy() - val_neg_emb.detach().cpu().numpy())
        # use sigmoid(dist_neg - dist_pos)
        diff = dist_neg - dist_pos
        diff = 1.0 / (1 + np.exp(-(diff) ))
        if diff >= confThreshold:
            return diff, True
        '''
        if (dist_neg - dist_pos) >= confThreshold:
            return True
        '''
        return diff, False


    def evaluationValid(self, validation_data, round, ifInTrainMap):
        validation_set = GateValidationTestDataset(validation_data)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=self.args.batch_size,
                                                           collate_fn=self.collate_fn)
        ndcgmetric, mrr, TP, FP, FN, acc, count = 0, 0, 0, 0, 0, 0, 0
        self.model.eval()
        index_seed = 0
        for local_batch, lengths in validation_generator:
            for attributes, length in zip(local_batch, lengths):
                attributes[0] = torch.tensor(attributes[0]).to(self.device)
                attr_emb = self.model.getAttrEmb(attributes[0])
                res = []
                for idx in range(1, length):
                    attributes[idx] = torch.tensor(attributes[idx]).to(self.device)
                    val_emb = self.model.getValueEmb(attributes[idx])
                    dist = np.linalg.norm(attr_emb.detach().numpy() - val_emb.detach().numpy())
                    res.append(dist)
                #prediction = [[tid, dist] for tid, dist in zip(attributes[1:], res)]
                prediction = np.argsort(res) + 1
                ground_truth = np.arange(length - 1) + 1

                ndcgmetric_s, mrr_s, TP_s, FP_s, FN_s, acc_s, count_s = metrics(prediction, ground_truth, attributes[0], ifInTrainMap, index_seed * 100)
                index_seed += 1
                ndcgmetric += ndcgmetric_s
                mrr += mrr_s
                TP += TP_s
                FP += FP_s
                FN += FN_s
                acc += acc_s
                count += count_s

                '''
                # NDCG
                pred = np.argsort(res) + 1
                ground_truth = np.arange(length - 1) + 1
                #ndcg += ndcg_score([ground_truth], [pred])
                ndcgmetric += retrieval_normalized_dcg(torch.tensor(pred, dtype=torch.float32),
                                                       torch.tensor(ground_truth))
                # MRR
                mrr_pred = [0] * len(pred)
                mrr_pred[np.argmin(pred)] = 1
                mrr_ground_truth = [True] + [False] * (len(pred) - 1)
                mrr_pred = torch.tensor(mrr_pred, dtype=torch.float32)
                mrr_ground_truth = torch.tensor(mrr_ground_truth)
                mrr += retrieval_reciprocal_rank(mrr_pred, mrr_ground_truth)

                # F-measure
                for pair in itertools.combinations(np.arange(len(pred)), 2):
                    if pair[0] == pair[1]:
                        continue
                    sc0, sc1 = pair
                    label, prediction = sc0 < sc1, pred[sc0] < pred[sc1]
                    # TP
                    if label == True and prediction == True:
                        TP += 1
                    # FP
                    if prediction == True and label == False:
                        FP += 1
                    # FN
                    if label == True and prediction == False:
                        FN += 1
                '''

        # evaluate metrics
        ndcg = ndcgmetric * 1.0 / (len(validation_generator) * self.args.batch_size)
        mrr = mrr * 1.0 / (len(validation_generator) * self.args.batch_size)
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        Fmeasure = 2 * precision * recall / (precision + recall)
        accuracy = acc * 1.0 / count #(len(validation_generator) * self.args.batch_size)

        logging.info("round={} ndcg={} mrr={} precision={} recall={} Fmeasure={}".format(round, ndcg, mrr, precision, recall, Fmeasure))
        print("round={} ndcg={} mrr={} precision={} recall={} Fmeasure={} accuracy={}".format(round, ndcg, mrr, precision, recall, Fmeasure, accuracy))



    def train(self, improved_data, training_data, validation_data, option):
        criterion = PairWiseLoss()
        if option == 'creatorna':
            criterion = PairWiseLossNA()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        training_set = GateDataset(improved_data, training_data, option)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=self.args.batch_size, shuffle=True)
        total_loss_min = sys.maxsize

        validation_set = GateValidationTestDataset(validation_data, option)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=self.args.batch_size,
                                                           collate_fn=self.collate_fn)

        for ep in range(self.args.epoch):
            total_loss = 0
            ndcg = 0
            ndcgmetric = 0
            mrr = 0
            start_time = time.time()
            self.model.train()
            accuracy = 0
            for local_batch in training_generator:
                self.model.zero_grad()
                local_batch = torch.stack(local_batch)
                local_batch = local_batch.to(self.device)
                res = self.model(local_batch)
                accuracy += ((res[1] - res[2]) > 0).sum().item()
                loss = criterion(res)
                total_loss += loss.item() / self.args.batch_size
                #print(total_loss)
                loss.backward()
                optimizer.step()
            end_time = time.time()
            epoch_time = (end_time - start_time)
            accuracy = accuracy * 1.0 / len(improved_data)
            logging.info('Epoch: ' + str(ep) + ' Total Loss: %.4f | execution time %.4f mins', total_loss, epoch_time)
            print('Epoch: {} Total Loss: {} Accuracy: {} | execution time {} seconds'.format(ep, total_loss, accuracy, epoch_time))

            # self.model.eval()
            # for local_batch, lengths in validation_generator:
            #     for attributes, length in zip(local_batch, lengths):
            #         attributes[0] = attributes[0].to(self.device)
            #         attr_emb = self.model.x_embed2last(self.model.x_input2embed(attributes[0]))
            #         res = []
            #         for idx in range(1, length):
            #             attributes[idx] = attributes[idx].to(self.device)
            #             val_emb = self.model.x_embed2last(self.model.x_input2embed(attributes[idx]))
            #             dist = np.linalg.norm(attr_emb.detach().numpy() - val_emb.detach().numpy())
            #             res.append(dist)
            #
            #         pred = np.argsort(res) + 1
            #         ground_truth = np.arange(length - 1) + 1
            #         ndcg += ndcg_score([ground_truth], [pred])
            #         ndcgmetric += retrieval_normalized_dcg(torch.tensor(pred, dtype=torch.float32),
            #                                                torch.tensor(ground_truth))
            #
            #         mrr_pred = [0] * len(pred)
            #         mrr_pred[np.argmin(pred)] = 1
            #         mrr_ground_truth = [True] + [False] * (len(pred) - 1)
            #         mrr_pred = torch.tensor(mrr_pred, dtype=torch.float32)
            #         mrr_ground_truth = torch.tensor(mrr_ground_truth)
            #         mrr += retrieval_reciprocal_rank(mrr_pred, mrr_ground_truth)
            #
            # logging.info(
            #     'NDCG score for validation set is ' + str(ndcg / (len(validation_generator) * self.args.batch_size)))
            #
            # logging.info(
            #     'NDCG score from torchmetrics for validation set is ' + str(
            #         ndcgmetric.item() / (len(validation_generator) * self.args.batch_size)))
            #
            # logging.info(
            #     'MRR score for validation set is ' + str(
            #         mrr.item() / (len(validation_generator) * self.args.batch_size)))

            if total_loss < total_loss_min:
                logging.info('Saving the best model')
                torch.save(self.model.state_dict(), 'best_saved_weights.pt')
                logging.info('Model saved as best_saved_weights.pt')
                total_loss_min = total_loss


# Here we define our NN module
class Net1(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Net1, self).__init__()
        self.x_input2embed = nn.Linear(input_dim, embed_dim)
        self.x_embed2last = nn.Linear(embed_dim, 2)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.dropout = nn.Dropout(0.1)

    def getAttrEmb(self, attr):
        return self.x_embed2last(self.x_input2embed(attr))

    def getValueEmb(self, value):
        return self.x_embed2last(self.x_input2embed(value))

    def forward(self, inputs_):
        dummy_reference_vector = self.x_input2embed(inputs_[0].float())
        dummy_reference_vector = self.x_embed2last(dummy_reference_vector)
        dummy_reference_vector = self.dropout(dummy_reference_vector)

        latest_value_vector = self.x_input2embed(inputs_[1].float())
        latest_value_vector = self.x_embed2last(latest_value_vector)
        latest_value_vector = self.dropout(latest_value_vector)

        non_latest_value_vector = self.x_input2embed(inputs_[2].float())
        non_latest_value_vector = self.x_embed2last(non_latest_value_vector)
        non_latest_value_vector = self.dropout(non_latest_value_vector)

        dummy_reference_vector = dummy_reference_vector.double()
        latest_value_vector = latest_value_vector.double()
        non_latest_value_vector = non_latest_value_vector.double()
        # https://discuss.pytorch.org/t/dot-product-batch-wise/9746

        # Adaptive Margin cosine similarity
        pos = self.cos(dummy_reference_vector, latest_value_vector)
        neg = self.cos(dummy_reference_vector, non_latest_value_vector)

        adaptive_margin = 1 - self.cos(latest_value_vector, non_latest_value_vector)

        return adaptive_margin, pos, neg


# Here we define our NN module
# two encoders that do not share parameters
class Net(nn.Module):
    def __init__(self, input_dim, embed_dim, final_dim=100):
        super(Net, self).__init__()
        self.x_input2embed_ = nn.Linear(input_dim, embed_dim)
        self.x_input2embed = nn.ReLU() #nn.Sigmoid()
        self.x_embed2last_ = nn.Linear(embed_dim, final_dim)
        #self.x_embed2last = nn.ReLU() #nn.Sigmoid()

        self.c_input2embed_ = nn.Linear(input_dim, embed_dim)
        self.c_input2embed = nn.ReLU() #nn.Sigmoid()
        self.c_embed2last_ = nn.Linear(embed_dim, final_dim)
        #self.c_embed2last = nn.ReLU() #nn.Sigmoid()

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.dropout = nn.Dropout(0.1)

    def getAttrEmb(self, attr):
        #return nn.functional.normalize(self.c_embed2last_(self.c_input2embed(self.c_input2embed_(attr))), dim=0)
        #return nn.functional.normalize(self.c_embed2last(self.c_embed2last_(self.c_input2embed(self.c_input2embed_(attr)))), dim=0)
        return self.c_embed2last_(self.c_input2embed(self.c_input2embed_(attr)))


    def getValueEmb(self, value):
        #return nn.functional.normalize(self.x_embed2last_(self.x_input2embed(self.x_input2embed_(value))), dim=0)
        #return nn.functional.normalize(self.x_embed2last(self.x_embed2last_(self.x_input2embed(self.x_input2embed_(value)))), dim=0)
        return self.x_embed2last_(self.x_input2embed(self.x_input2embed_(value)))

    def forward(self, inputs_):
        dummy_reference_vector = self.c_input2embed_(inputs_[0].float())
        dummy_reference_vector = self.c_input2embed(dummy_reference_vector)
        dummy_reference_vector = self.c_embed2last_(dummy_reference_vector)
        #dummy_reference_vector = self.c_embed2last(dummy_reference_vector)
        #dummy_reference_vector = self.dropout(dummy_reference_vector)

        latest_value_vector = self.x_input2embed_(inputs_[1].float())
        latest_value_vector = self.x_input2embed(latest_value_vector)
        latest_value_vector = self.x_embed2last_(latest_value_vector)
        #latest_value_vector = self.x_embed2last(latest_value_vector)
        #latest_value_vector = self.dropout(latest_value_vector)

        non_latest_value_vector = self.x_input2embed_(inputs_[2].float())
        non_latest_value_vector = self.x_input2embed(non_latest_value_vector)
        non_latest_value_vector = self.x_embed2last_(non_latest_value_vector)
        #non_latest_value_vector = self.x_embed2last(non_latest_value_vector)
        #non_latest_value_vector = self.dropout(non_latest_value_vector)

        dummy_reference_vector = dummy_reference_vector.double()
        latest_value_vector = latest_value_vector.double()
        non_latest_value_vector = non_latest_value_vector.double()
        # https://discuss.pytorch.org/t/dot-product-batch-wise/9746

        # Adaptive Margin cosine similarity
        #pos = self.cos(dummy_reference_vector, latest_value_vector)
        #neg = self.cos(dummy_reference_vector, non_latest_value_vector)

        # L2 distance, ot dot product
        pos = nn.functional.tanh( (dummy_reference_vector * latest_value_vector).sum(1) ) #(dummy_reference_vector - latest_value_vector).pow(2).sum(1).sqrt()
        neg = nn.functional.tanh( (dummy_reference_vector * non_latest_value_vector).sum(1) ) #(dummy_reference_vector - non_latest_value_vector).pow(2).sum(1).sqrt()

        adaptive_margin = 1 + self.cos(latest_value_vector, non_latest_value_vector)

        return adaptive_margin, pos, neg

class PairWiseLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PairWiseLoss, self).__init__()

    def forward(self, res):
        adaptive_margin, pos, neg = res
        #return torch.sum(torch.exp(torch.max(torch.tensor(0), (adaptive_margin - pos)))) + torch.sum(torch.exp(torch.max(torch.tensor(0), (adaptive_margin + neg))))
        return torch.sum(torch.max(-pos + neg + adaptive_margin, torch.tensor(0)))
        #return torch.sum(torch.max(pos - neg + adaptive_margin, torch.tensor(0)))

class PairWiseLossNA(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PairWiseLossNA, self).__init__()

    def forward(self, res):
        adaptive_margin, pos, neg = res
        adaptive_margin = 0
        return torch.sum(torch.exp(- pos)) + torch.sum(torch.exp(neg))
