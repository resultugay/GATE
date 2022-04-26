import logging

from sklearn.metrics import ndcg_score

from creator.CreatorFactory import CreatorFactory
import torch
import pickle
import numpy as np
import torch.nn as nn
import random
import pandas as pd
from critic.Critic import Critic
from creator.BERT import get_emb
from creator.GateDataset import GateValidationTestDataset
from torchmetrics.functional import retrieval_reciprocal_rank, retrieval_normalized_dcg


class Gate:
    _instance = None
    # training_data_embedded is the data that we embedded
    # all the attribute values with BERT, this is a list and each element
    # in the list consists of a tuple with three tensors with 1536 dimension
    # The first element in the tuple is the dummy attribute vector
    # Second element is the vector of latest attribute value
    # Third element is the vector of non-latest attribute value
    #training_embedded = None

    # training_processed is processed version of the training data
    # This is a list of lists, each list consists of three elements
    # [attr, (latest_attr_idx,latest_attr), (non_latest_attr_idx, non_latest_attr)]
    # These indexes help us to find the context vector of the tuple
    #training_processed = None
    # Sentence embeddings of the tuples
    #training_sentence_embeddings = None
    # Unique attribute embeddings of the attribute values
    #training_attribute_embeddings = {}
    # This is training data
    training_data = None
    simple_ccs = None
    complex_ccs = None
    training = dict()
    validation = dict()
    test = dict()
    complex_ccs = []
    simple_ccs = {}

    def __new__(cls):
        if cls._instance is None:
            logging.info('Creating the GATE object')
            cls._instance = super(Gate, cls).__new__(cls)
        else:
            logging.info('GATE object already exist')
        return cls._instance

    def load_data(self):
        # Training data
        try:
            self.training['training_embedded'] = torch.load(self.args.data + 'training_embedded.pt')
        except Exception as e:
            self.training['training_embedded'] = None
            logging.info('Training Embedded file missing, embeddings will be fetched during training')

        with open(self.args.data + 'training_processed.pkl', 'rb') as f:
            self.training['data_processed'] = pickle.load(f)

        with open(self.args.data + 'training_attribute_embeddings.pkl', 'rb') as f:
            self.training['data_attribute_embeddings'] = pickle.load(f)

        self.training['data_sentence_embeddings'] = torch.load(self.args.data + 'training_sentence_embeddings.pt')
        self.training['data'] = pd.read_csv(self.args.data + 'training.csv')

        # Validation data
        with open(self.args.data + 'validation_processed.pkl', 'rb') as f:
            self.validation['data_processed'] = pickle.load(f)

        self.validation['data_attribute_embeddings'] = dict()

        with open(self.args.data + 'validation_attribute_embeddings.pkl', 'rb') as f:
            self.validation['data_attribute_embeddings'] = pickle.load(f)

        self.validation['data_sentence_embeddings'] = torch.load(self.args.data + 'validation_sentence_embeddings.pt')
        self.validation['data'] = pd.read_csv(self.args.data + 'validation.csv')

        # Test data
        with open(self.args.data + 'test_processed.pkl', 'rb') as f:
            self.test['data_processed'] = pickle.load(f)

        self.test['data_attribute_embeddings'] = dict()

        with open(self.args.data + 'test_attribute_embeddings.pkl', 'rb') as f:
            self.test['data_attribute_embeddings'] = pickle.load(f)

        self.test['data_sentence_embeddings'] = torch.load(self.args.data + 'test_sentence_embeddings.pt')
        self.test['data'] = pd.read_csv(self.args.data + 'test.csv')

    def initialize(self, args):
        self.args = args
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('GATE initializing')
        logging.info('Data ' + self.args.data)
        logging.info('Embedding Dim : ' + str(self.args.embedding_dim))
        logging.info('Epoch : ' + str(self.args.epoch))
        logging.info('Learning Rate : ' + str(self.args.lr))
        logging.info('Batch Size : ' + str(self.args.batch_size))
        logging.info('High Confidence Sample Ratio : ' + str(self.args.high_conf_sample_ratio))
        logging.info('Data loading')
        self.load_data()
        self.creator = CreatorFactory.get_creator(args)
        self.critic = Critic()
        self.complex_ccs, self.simple_ccs = self.read_ccs(self.args.data)

        logging.info('Data loaded')

        logging.info('GATE initialized')

    def read_ccs(self, path):
        try:
            with open(path + 'ccs.txt') as f:
                for line in f:
                    cc = line.split(',')
                    if len(cc) > 2:
                        if cc[0] not in self.simple_ccs:
                            self.simple_ccs[cc[0]] = []
                        self.simple_ccs[cc[0]].append((str(cc[1]).strip(), str(cc[2]).strip()))
                    else:
                        cc = line.split('>')
                        self.complex_ccs.append((str(cc[0]).strip(), str(cc[1].strip())))
        except:
            pass
        return self.complex_ccs, self.simple_ccs

    def train(self):
        logging.info('Training Started')
        improved_data = self.training['training_embedded'] if self.training['training_embedded'] else [1]
        improved_data_processed = self.training['data_processed']
        round = 1
        logging.info('Size of initial training data ' + str(len(self.training['data_processed'])))
        while improved_data:
            logging.info('Training round ' + str(round) + ' started')
            self.creator.train(improved_data, self.training, self.validation)
            high_conf_ccs = self.choose_high_confidence(self.creator.model, improved_data_processed, self.args.high_conf_sample_ratio)
            new_temporal_orders = self.critic.deduce(high_conf_ccs, self.training_data, self.complex_ccs)
            conflicted_orders = self.critic.conflict(self.simple_ccs, high_conf_ccs)
            improved_data, improved_data_processed = self.create_training_data(new_temporal_orders, conflicted_orders)
            logging.info('Training round ' + str(round) + ' finished ' + str(
                len(improved_data)) + ' new training instances found')
            round += 1
        logging.info('Training Finished')

    def evaluate(self):
        logging.info('Evaluation Started')
        test_set = GateValidationTestDataset(self.test)
        test_generator = torch.utils.data.DataLoader(test_set, batch_size=self.args.batch_size,
                                                     collate_fn=self.creator.collate_fn)
        logging.info('Size of training data ' + str(len(test_generator)))

        if self.creator.model is None:
            from creator.GateCreator import Net
            model = Net(self.args.input_dim, self.args.embedding_dim)
            path = 'best_saved_weights.pt'
            model.load_state_dict(torch.load(path))

        self.creator.model.eval()
        ndcg = 0
        ndcgmetric = 0
        mrr = 0
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        for local_batch, lengths in test_generator:
            for attributes, length in zip(local_batch, lengths):
                attributes[0] = attributes[0].to(device)
                attr_emb = self.creator.model.x_embed2last(self.creator.model.x_input2embed(attributes[0]))
                res = []
                for idx in range(1, length):
                    attributes[idx] = attributes[idx].to(device)
                    val_emb = self.creator.model.x_embed2last(self.creator.model.x_input2embed(attributes[idx]))
                    dist = np.linalg.norm(attr_emb.detach().numpy() - val_emb.detach().numpy())
                    res.append(dist)

                pred = np.argsort(res) + 1
                ground_truth = np.arange(length - 1) + 1
                ndcg += ndcg_score([ground_truth], [pred])
                ndcgmetric += retrieval_normalized_dcg(torch.tensor(pred, dtype=torch.float32),
                                                       torch.tensor(ground_truth))

                mrr_pred = [0] * len(pred)
                mrr_pred[np.argmin(pred)] = 1
                mrr_ground_truth = [True] + [False] * (len(pred) - 1)
                mrr += retrieval_reciprocal_rank(torch.tensor(mrr_pred, dtype=torch.float32), torch.tensor(mrr_ground_truth))

        logging.info('NDCG score for test set is ' + str(ndcg / (len(test_generator) * self.args.batch_size)))

        logging.info(
            'NDCG score from torchmetrics for validation set is ' + str(
                ndcgmetric.item() / (len(test_generator) * self.args.batch_size)))

        logging.info('MRR score for test set is ' + str(mrr.item() / (len(test_generator) * self.args.batch_size)))

    def choose_high_confidence(self, model, training_processed, high_conf_sample_ratio):
        logging.info('Choosing high confidence temporal orders')
        if self.creator.model is None:
            from creator.GateCreator import Net
            model = Net(self.args.input_dim, self.args.embedding_dim)
            path = 'best_saved_weights.pt'
            model.load_state_dict(torch.load(path))

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        confidence_map = {}

        if isinstance(training_processed, dict):
            embeddings = self.get_embeddings(training_processed)
            cnt = 0
            for attribute, orders in training_processed.items():
                for pos_att, neg_att in orders:
                    attr_vec, pos_vec, neg_vec = embeddings[cnt]

                    pos_sim = cos(attr_vec, pos_vec)
                    neg_sim = cos(attr_vec, neg_vec)
                    if pos_sim > neg_sim:
                        if (attribute, pos_att, neg_att) not in confidence_map:
                            confidence_map[(attribute, pos_att, neg_att)] = 0
                        confidence_map[(attribute, pos_att, neg_att)] += 1
                    cnt += 1
        else:
            # sample training data, reduce calculations
            for index, i in random.sample(list(enumerate(training_processed)), (len(training_processed) * high_conf_sample_ratio)):
                attribute = i[0]
                pos_context_index, pos_att = i[1]
                neg_context_index, neg_att = i[2]

                attribute_emb = self.training['data_attribute_embeddings'][attribute]
                pos_context_emb = self.training['data_sentence_embeddings'][pos_context_index]
                pos_att_emb = self.training['data_attribute_embeddings'][str(pos_att)]
                neg_context_emb = self.training['data_sentence_embeddings'][neg_context_index]
                neg_att_emb = self.training['data_attribute_embeddings'][str(neg_att)]

                # concat all of the tensors
                attribute_emb = torch.cat((attribute_emb, attribute_emb), 0)
                pos_instance = torch.cat((pos_context_emb, pos_att_emb), 0)
                neg_instance = torch.cat((neg_context_emb, neg_att_emb), 0)

                attr_vec = model.x_embed2last(model.x_input2embed(attribute_emb))
                pos_vec = model.x_embed2last(model.x_input2embed(pos_instance))
                neg_vec = model.x_embed2last(model.x_input2embed(neg_instance))

                pos_sim = cos(attr_vec, pos_vec)
                neg_sim = cos(attr_vec, neg_vec)
                if pos_sim > neg_sim:
                    if (attribute, pos_att, neg_att) not in confidence_map:
                        confidence_map[(attribute, pos_att, neg_att)] = 0
                    confidence_map[(attribute, pos_att, neg_att)] += 1

        confidence_map = dict(sorted(confidence_map.items(), key=lambda x: x[1], reverse=True))
        high_conf_ccs = {}
        for i in list(confidence_map.keys())[:100]:
            if i[0] not in high_conf_ccs:
                high_conf_ccs[i[0]] = []
            high_conf_ccs[i[0]].append((i[1], i[2]))
        return high_conf_ccs

    def create_training_data(self, new_temporal_orders, conflicted_orders):
        data = self.get_embeddings(new_temporal_orders)
        data += self.get_embeddings(conflicted_orders)
        return data, {**new_temporal_orders, **conflicted_orders}

    def get_embeddings(self, order_dict):
        data = []
        for attribute, orders in order_dict.items():
            if attribute not in self.training['data_attribute_embeddings']:
                attribute_emb = get_emb(attribute)
                self.training['data_attribute_embeddings'][attribute] = attribute_emb

            attribute_emb = self.training['data_attribute_embeddings'][attribute]
            attribute_emb = torch.cat((attribute_emb, attribute_emb), 0)

            for pos_att, neg_att in orders:
                if pos_att not in self.training['data_attribute_embeddings']:
                    pos_att_emb = get_emb(str(pos_att))
                    self.training['data_attribute_embeddings'][pos_att] = pos_att_emb

                if neg_att not in self.training['data_attribute_embeddings']:
                    neg_att_emb = get_emb(str(neg_att))
                    self.training['data_attribute_embeddings'][neg_att] = neg_att_emb

                pos_att_emb = self.training['data_attribute_embeddings'][str(pos_att)]
                neg_att_emb = self.training['data_attribute_embeddings'][str(neg_att)]
                pos_att_emb = torch.cat((pos_att_emb, pos_att_emb), 0)
                neg_att_emb = torch.cat((neg_att_emb, neg_att_emb), 0)
                data.append((attribute_emb, pos_att_emb, neg_att_emb))
        return data
