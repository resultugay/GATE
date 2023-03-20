import logging

from sklearn.metrics import ndcg_score

from creator.CreatorFactory import CreatorFactory
import torch
import pickle
import numpy as np
import torch.nn as nn
import random
import itertools
import time
import copy
import pandas as pd
from collections import defaultdict
from critic.Critic import Critic
from critic.Fixes import *
from critic.GS import GlobalStructures
from critic.Valuation import *
from creator.BERT import get_emb
from creator.GateDataset import GateValidationTestDataset
from metrics import metrics, isTrainingKey, FScore
from utility import procComm_
from torchmetrics.functional import retrieval_reciprocal_rank, retrieval_normalized_dcg

DELIMITOR = '---'

MAX_ROUND = 15

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
    simple_ccs = None
    complex_ccs = None
    training = dict()
    validation = dict()
    test = dict()
    complex_ccs = []
    simple_ccs = {}
    schemas = None

    dataProcessedMap = defaultdict()
    ccs = None
    # other parameters
    ifIncrementalLearning = True

    # if acc in validationT does not increase in terminT rounds, then terminate
    terminT = 3


    def __new__(cls):
        if cls._instance is None:
            logging.info('Creating the GATE object')
            cls._instance = super(Gate, cls).__new__(cls)
        else:
            logging.info('GATE object already exist')
        return cls._instance

    def load_data_new(self):
        # Training data
        try:
            self.training['training_embedded'] = torch.load(self.args.data + 'training_embedded.pt')
        except Exception as e:
            self.training['training_embedded'] = None
            logging.info('Training Embedded file missing, embeddings will be fetched during training')

        with open(self.args.data + 'training_processed.pkl', 'rb') as f:
            self.training['data_processed'] = pickle.load(f)
        with open(self.args.data + 'validation_processed.pkl', 'rb') as f:
            self.validation['data_processed'] = pickle.load(f)
        with open(self.args.data + 'testing_processed.pkl', 'rb') as f:
            self.test['data_processed'] = pickle.load(f)

        self.usedEntitiesMap = defaultdict()
        for data in self.training['data_processed']:
            self.usedEntitiesMap[data[-1]] = 0


        # construct index for validation
        for pid, processed in enumerate(self.validation['data_processed']):
            attr, _list = processed[0], processed[1]
            for i, e in enumerate(_list):
                self.dataProcessedMap[str(attr) + DELIMITOR + str(e[0])] = [pid, i]


        # few-shot, sampling 10% training data
        np.random.seed(30)
        sample_ratio = 0.5
        numT = len(self.training['data_processed'])
        sc = np.random.choice(len(self.training['data_processed']), int(sample_ratio * len(self.training['data_processed'])), replace=False)
        _list_processed, _list_embedded = [], []
        for e in sc:
            ll = self.training['data_processed'][e]
            # only keep temporal orders such that
            if ll[1][0] <= ll[2][0]:
                continue
            _list_processed.append(self.training['data_processed'][e])
            _list_embedded.append(self.training['training_embedded'][e])

        # validationForTrain: used for terminate the iteration
        sc_validT = [e for e in range(numT) if e not in sc]
        self.validationT = dict()
        _list_processed_validT, _list_embedded_validT = [], []
        for e in sc_validT:
            ll = self.training['data_processed'][e]
            if ll[1][0] <= ll[2][0]:
                continue
            _list_processed_validT.append(self.training['data_processed'][e])
            _list_embedded_validT.append(self.training['training_embedded'][e])

        self.training['data_processed'] = _list_processed
        self.training['training_embedded'] = _list_embedded
        self.validationT['data_processed'] = _list_processed_validT
        self.validationT['training_embedded'] = _list_embedded_validT

        # varying Gamma
        sc = np.arange(len(self.training['data_processed']))
        np.random.seed(42)
        np.random.shuffle(sc)
        num = int(self.args.gamma * len(sc))
        sc = sc[:num]
        _list_processed, _list_embedded = [], []
        for e in sc:
            _list_processed.append(self.training['data_processed'][e])
            _list_embedded.append(self.training['training_embedded'][e])
        self.training['data_processed'] = _list_processed
        self.training['training_embedded'] = _list_embedded


        # varying entity ratio
        entities_list = list(set([e[-1] for e in self.validation['data_processed']]))
        np.random.seed(42)
        np.random.shuffle(entities_list)
        num = int(self.args.entityRatio * len(entities_list))
        #self.usedEntitiesMap = defaultdict()
        for eid in entities_list[:num]:
            self.usedEntitiesMap[eid] = 0

        # make sure that the list contains at least two elements
        temp_validation_processed = []
        for record in self.validation['data_processed']:
            if len(record[1]) <= 1:
                continue
            temp_validation_processed.append(record)
        self.validation['data_processed'] = temp_validation_processed

        # varying D_T
        if self.args.D_T != 1.0:
            sc = np.arange(len(self.validation['data_processed']))
            np.random.seed(42)
            np.random.shuffle(sc)
            num = int(self.args.D_T * len(sc))
            sc = sc[:num]
            _list_processed = []
            for e in sc:
                _list_processed.append(self.validation['data_processed'][e])
            self.validation['data_processed'] = _list_processed


        # record all training instances, and exclude them in validation and testing data
        self.ifInTrainMap = defaultdict()
        for record in self.training['data_processed']:
            attr, t0, t1 = record[0], record[1][0], record[2][0]
            key = isTrainingKey(t0, t1, attr)
            key_ = isTrainingKey(t1, t0, attr)
            self.ifInTrainMap[key] = 0
            self.ifInTrainMap[key_] = 0

        attribute_embeddings = dict()
        with open(self.args.data + 'data_attribute_embeddings.pkl', 'rb') as f:
            attribute_embeddings = pickle.load(f)

        # transfer attribute embeddings to tensor
        for k in attribute_embeddings.keys():
            attribute_embeddings[k] = torch.tensor(torch.nan_to_num(attribute_embeddings[k]), dtype=torch.float32) #torch.tensor(attribute_embeddings[k])

        sentence_embeddings = torch.load(self.args.data + 'data_sentence_embeddings.pt')
        sentence_embeddings = torch.tensor(torch.nan_to_num(sentence_embeddings), dtype=torch.float32)
        # check nan
        for i in range(len(self.training['training_embedded'])):
            self.training['training_embedded'][i][0] = torch.nan_to_num(self.training['training_embedded'][i][0])
            self.training['training_embedded'][i][1] = torch.nan_to_num(self.training['training_embedded'][i][1])
            self.training['training_embedded'][i][2] = torch.nan_to_num(self.training['training_embedded'][i][2])

        # data = pd.read_csv(self.args.data + 'data.csv')
        # specially handle comm_
        if self.args.data.find('comm_') != -1 or self.args.data.find('comm') != -1:
            data = procComm_(self.args.data + 'data.csv')
        else:
            data = pd.read_csv(self.args.data + 'data.csv')

        self.training['data_attribute_embeddings'] = attribute_embeddings
        self.training['data_sentence_embeddings'] = sentence_embeddings
        self.training['data'] = data

        self.validation['data_attribute_embeddings'] = attribute_embeddings
        self.validation['data_sentence_embeddings'] = sentence_embeddings
        self.validation['data'] = data

        self.test['data_attribute_embeddings'] = attribute_embeddings
        self.test['data_sentence_embeddings'] = sentence_embeddings
        self.test['data'] = data

        # timeliness attributes
        self.timelinessAttrs = set()
        for e in self.training['data_processed']:
            self.timelinessAttrs.add(e[0])
        for e in self.validation['data_processed']:
            self.timelinessAttrs.add(e[0])
        for e in self.test['data_processed']:
            self.timelinessAttrs.add(e[0])

        self.timelinessAttrs = [str(e) for e in self.timelinessAttrs]
        self.data = np.array(data.values)
        self.validationTIDs = defaultdict(int)
        for info in self.validation['data_processed']:
            for e in info[1]:
                self.validationTIDs[e[0]] = 0

        self.schemas = list(data.columns)
        self.attrsMap = defaultdict()
        for sid, s in enumerate(self.schemas):
            self.attrsMap[s] = sid

        # only for test variant algorithms, i.e., creatornc and creatorne
        if self.args.variants == 'creatornc':
            for i in range(len(self.training['training_embedded'])):
                embedds_list = self.training['training_embedded'][i]
                for j in range(len(embedds_list)):
                    half_dims = int(len(self.training['training_embedded'][i][j]) / 2)
                    self.training['training_embedded'][i][j][half_dims:] = self.training['training_embedded'][i][j][:half_dims]
        if self.args.variants == 'creatorne':
            # randomly generate embeddings
            for i in range(len(self.training['data_sentence_embeddings'])):
                _dim = len(self.training['data_sentence_embeddings'][i])
                self.training['data_sentence_embeddings'][i] = torch.tensor(np.random.random(_dim), dtype=torch.float32)
            for k, v in self.training['data_attribute_embeddings'].items():
                _dim = len(v)
                self.training['data_attribute_embeddings'][k] = torch.tensor(np.random.random(_dim), dtype=torch.float32)
            
            tmp_train_embedded = []
            attribute_embeddings = self.training['data_attribute_embeddings']
            sentence_embeddings = self.training['data_sentence_embeddings']
            for index, values in enumerate(self.training['data_processed']):
                attribute = values[0]
                print('index : ', index)
                pos_context_index, pos_att = values[1][0], str(values[1][1])
                neg_context_index, neg_att = values[2][0], str(values[2][1])

                attribute_emb = attribute_embeddings[attribute]
                pos_context_emb = sentence_embeddings[pos_context_index]
                pos_att_emb = attribute_embeddings[pos_att]
                neg_context_emb = sentence_embeddings[neg_context_index]
                neg_att_emb = attribute_embeddings[neg_att]

                attribute_emb = torch.cat([attribute_emb, attribute_emb], 0)
                pos_instance = torch.cat([pos_context_emb, pos_att_emb], 0)
                neg_instance = torch.cat([neg_context_emb, neg_att_emb], 0)
                tmp_train_embedded.append([attribute_emb, pos_instance, neg_instance])
            self.training['training_embedded'] = tmp_train_embedded



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
        #self.load_data()
        self.load_data_new()
        self.creator = CreatorFactory.get_creator(args)
        self.critic = Critic(self.data, self.schemas, self.timelinessAttrs, self.attrsMap)
        # self.complex_ccs, self.simple_ccs = self.read_ccs(self.args.data)
        self.ccs = self.load_ccs()

        logging.info('Data loaded')

        logging.info('GATE initialized')

    def load_ccs(self):
        path = self.args.data + self.args.ccs #'CCs.txt'
        print('CCs PATH : ', path)
        ccs = None
        try:
            rulesStr = []
            with open(path) as f:
                for line in f:
                    # filter rules that already exists in the training instances
                    if line.find('data.t0.id') != -1 and line.find('data.t1.id') != -1:
                        continue
                    print('read ...')
                    rulesStr.append(line.strip())
            ccs = CCs(rulesStr, self.timelinessAttrs)
        except:
            pass
        return ccs

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
            new_temporal_orders = self.critic.deduce(high_conf_ccs, self.training['data'], self.complex_ccs)
            conflicted_orders = self.critic.conflict(self.simple_ccs, high_conf_ccs)
            improved_data, improved_data_processed = self.create_training_data(new_temporal_orders, conflicted_orders)
            logging.info('Training round ' + str(round) + ' finished ' + str(
                len(improved_data)) + ' new training instances found')
            round += 1
        logging.info('Training Finished')


    def transformTOs(self, dataProcessed):
        '''
        Transform data_processed to a list of TemporalOrder
        :param dataProcessed: a list of [attribute, (tid_0, value_0), (tid_1, value_1)]
        :return:
        '''
        tos = []
        for attr, pos, neg, eid in dataProcessed:
            to = TemporalOrder(pos[0], neg[0], attr, Operator.GREATOR_THAN, eid, 1.0)
            tos.append(to)
        return tos

    def updateTrainingData(self, improved_data, training, data_aug_processed, data_aug_embedds, data_aug_tos):
        '''
        update improved_data, training_embedded, and data_processed
        :param data_aug_embedds: a list of [attr_vec, pos_vec, neg_vec]
                data_aug_tos: a list of [attribute, (tid_0, value_0), (tid_1, value_1)]
        :return:
        '''
        improved_data += data_aug_embedds
        training['data_processed'] += data_aug_processed
        '''
        # transform to Temporal Orders
        tos = data_aug_tos #self.transformTOs(data_aug_tos)
        for to in tos:
            GS.addTO(to)
        '''

    def stableTOKey(self, t_0, t_1, attr, eid):
        #return str(t_0) + '---' + str(t_1) + '---' + str(attr)
        return str(t_0) + ETID_ATTR_DELIMITOR + str(t_1) + ETID_ATTR_DELIMITOR + str(attr) + '_' + str(eid)

    def addstableTOsKeyMap(self, stableTOsMap, tos):
        for to in tos:
            stableTOsMap[self.stableTOKey(to.t0, to.t1, to.attribute, to.eid)] = 1


    def initialAddTrainingData(self):
        # only consider CCs: t_0.A = XXX ^ t_1.A = XXX -> t_0.A < t_1.A (temporal order)
        
        def selectCCsSpecial(cc):
            attr, val_1, val_2 = None, None, None
            lhs = cc.getLHSs()
            rhs = cc.getRHS()
            ifSpecial = True
            if len(lhs) != 2:
                ifSpecial = False
            for p in lhs:
                if not p.isConstant():
                    ifSpecial = False
                    break
            if not rhs.isTimelinessPredicate():
                ifSpecial = False
            if not (rhs.index1 == 't0' and rhs.index2 == 't1' and rhs.operator == '<'): # only support t0.A < t1.A
                ifSpecial = False
            if ifSpecial:
                cnt_1 = [lhs[0].index1, lhs[0].constant]
                cnt_2 = [lhs[1].index1, lhs[1].constant]
                attr = rhs.operand1['attribute']
                if cnt_1[0] == 't0' and cnt_2[0] == 't1':
                    val_1, val_2 = cnt_1[1], cnt_2[1]
                else:
                    val_1, val_2 = cnt_2[1], cnt_1[1]
            return attr, val_1, val_2, ifSpecial

        temporal_orders_list = []
        for cc in self.ccs.CCsList:
            attr, val_1, val_2, ifSpecial = selectCCsSpecial(cc)
            if ifSpecial:
                temporal_orders_list.append([attr, val_1, val_2])   # val_1 < val_2
        print('The number of temporal orders is {}'.format(len(temporal_orders_list)))
        print(temporal_orders_list)
        # store training data with format (attr<--->tid0<--->tid1) that tid0 > tid1
        limitor_init = '<--->'
        temporal_orders_map = defaultdict()
        for to in temporal_orders_list:
            temporal_orders_map[str(to[0]) + limitor_init + str(to[2]) + limitor_init + str(to[1])] = 1
        trainUsedMap = defaultdict()
        for index, record in enumerate(self.training['data_processed']):
            attr, eid = record[0], record[-1]
            kk = str(attr) + limitor_init + str(record[1][0]) + limitor_init + str(record[2][0])
            trainUsedMap[kk] = 1
        init_more_data_processed = []
        for index, record in enumerate(self.validation['data_processed']):
            attr, eid = record[0], record[-1]
            for sc_pair in itertools.permutations(np.arange(len(record[1])), 2):
                r_1, r_2 = record[1][sc_pair[0]], record[1][sc_pair[1]]
                kk = str(attr) + limitor_init + str(r_1[1]) + limitor_init + str(r_2[1])
                kk_ = str(attr) + limitor_init + str(r_1[0]) + limitor_init + str(r_2[0])
                if kk in temporal_orders_map:
                    print('count : ', kk)
                    if (kk_ not in trainUsedMap):
                        init_more_data_processed.append([attr, r_1, r_2, eid])
        # data embeddings
        init_more_data_embedds = []
        attribute_embeddings = self.validation['data_attribute_embeddings']
        sentence_embeddings = self.validation['data_sentence_embeddings']
        for index, values in enumerate(init_more_data_processed):
            attribute = values[0]
            pos_context_index, pos_att = values[1][0], str(values[1][1])
            neg_context_index, neg_att = values[2][0], str(values[2][1])
            attribute_emb = attribute_embeddings[attribute]
            pos_context_emb = sentence_embeddings[pos_context_index]
            pos_att_emb = attribute_embeddings[pos_att]
            neg_context_emb = sentence_embeddings[neg_context_index]
            neg_att_emb = attribute_embeddings[neg_att]

            attribute_emb = torch.cat([attribute_emb, attribute_emb], 0)
            pos_instance = torch.cat([pos_context_emb, pos_att_emb], 0)
            neg_instance = torch.cat([neg_context_emb, neg_att_emb], 0)
            init_more_data_embedds.append([attribute_emb, pos_instance, neg_instance])
        return init_more_data_processed, init_more_data_embedds
        

    def train_(self):
        '''
        GATE overflow, iteratively adopt creator and critic
        :return:
        '''
        accValidT, countTerminT = 0, 0
        overall_time = 0

        creator_start_time = time.time()
        logging.info('Training Started')
        improved_data = self.training['training_embedded'] if self.training['training_embedded'] else [1]
        improved_data_processed = self.training['data_processed']
        GS = GlobalStructures()
        if self.args.variants != 'critic':
            round = -1
            self.creator.train(improved_data, self.training, self.validation, self.args.variants)
            train_time = time.time() - creator_start_time
            self.evaluationValid_new(self.validation, GS, round, self.ifInTrainMap)
        if self.args.variants != 'gate' and self.args.variants != 'gatenc' and self.args.variants != 'creatoritr':
            print('The overall time for all rounds is {} and the training time is {}'.format(time.time() - creator_start_time, train_time))
            return
        
        # add more training data generated by special CCs
        init_more_data_processed, init_more_data_embedds = self.initialAddTrainingData()
        print('More number of data is {}'.format(len(init_more_data_processed)))
        self.training['training_embedded'] += init_more_data_embedds
        self.training['data_processed'] += init_more_data_processed

        improved_data = self.training['training_embedded'] if self.training['training_embedded'] else [1]
        improved_data_processed = self.training['data_processed']

        round = 0
        logging.info('Size of initial training data ' + str(len(self.training['data_processed'])))
        GS, deltaTOStable, deltaTOML, indicator = GlobalStructures(), self.transformTOs(improved_data_processed), [], True
        # set the initial stable set of temporal orders
        GS.addTOs(deltaTOStable)
        #stableTOsMap = defaultdict()         # store all tuple pairs of temporal orders
        #self.addstableTOsKeyMap(stableTOsMap, deltaTOStable)
        data_aug_embedds, data_aug_processed, data_aug_tos = [], [], []         # new stable data
        while True:
            start_overall_time_round = time.time()
            logging.info('Training round ' + str(round) + ' started')
            # -1. add stable temporal orders
            start_time = time.time()
            self.updateTrainingData(improved_data, self.training, data_aug_processed, data_aug_embedds, data_aug_tos)
            if round > 0:
                deltaTOStable = data_aug_tos #self.transformTOs(data_aug_tos)
            print("-1. The time of update training data is ", time.time() - start_time)

            # 1. check the catastrophic forgetting issue
            start_time = time.time()
            if round > 0 and self.args.ifIncrementalLearning:
                mispredicted_improved_data = self.creator.evaluate(improved_data, self.training)
            else:
                mispredicted_improved_data = improved_data
            print("1. The time of evaluating training data is {} with {} number of training data".format(time.time() - start_time, len(improved_data)))
            # 2. prepare the training data
            training_data = mispredicted_improved_data + data_aug_embedds
            # 3. incrementally train
            #self.creator.train(improved_data, self.training, self.validation)
            start_time = time.time()
            if self.args.variants != 'critic':
                self.creator.train(training_data, self.training, self.validation, self.args.variants)

            print("2. The time of training Creator is {} with {} number of training data ".format(time.time() - start_time, len(training_data)))

            # check variants
            if self.args.variants != 'gate' and self.args.variants != 'gatenc' and self.args.variants != 'creatoritr':
                break

            # # 4. add stable temporal orders
            # start_time = time.time()
            # self.updateTrainingData(improved_data, self.training, data_aug_processed, data_aug_embedds, data_aug_tos)
            # if round > 0:
            #     deltaTOStable = data_aug_tos #self.transformTOs(data_aug_tos)
            # print("3. The time of update training data is ", time.time() - start_time)
            # 5. retrieve all temporal orders with high confidence by Creator
            #deltaTOML = self.choose_high_confidence_(self.validation, self.args.high_conf_sample_ratio, self.args.conf_threshold, stableTOsMap)
            start_time = time.time()
            deltaTOML = []
            if self.args.variants == 'creatoritr' or (self.args.variants != 'critic' and round > 0): 
                ''' Here do not use ML inference of conf to deduce temporal orders, and make sure all deduced temporal orders are correct
                    as long as CCS and Gamma are correct.
                '''
                deltaTOML = self.choose_high_confidence_(self.validation, self.args.high_conf_sample_ratio, self.args.conf_threshold,
                                                         GS.stableTOs, min(int(0.1 * len(improved_data)), self.args.maxMLData))
            print("4. The time of choosing high confidence TOs is {} with {} temporal orders".format(time.time() - start_time, len(deltaTOML)))
            #new_temporal_orders = self.critic.deduce(high_conf_ccs, self.training['data'], self.complex_ccs)
            # new_temporal_orders are deduced by deltaTOStable, not deltaTOML
            start_time = time.time()
            if self.args.variants != 'creatoritr':
                new_temporal_orders, GS, indicator = self.critic.deduce(self.ccs, deltaTOStable, deltaTOML, indicator, GS, self.args.variants, round)
                # new_temporal_orders, GS, indicator = self.critic.deduce(self.ccs, deltaTOStable, deltaTOML, indicator, GS, self.args.variants, -1)
            else:
                new_temporal_orders, GS, indicator = [], GS, True
            print("5. The time of Chase is {} with {} temporal orders".format(time.time() - start_time, len(new_temporal_orders)))
            # only keep temporal orders in validation data
            new_temporal_orders_ = defaultdict()
            for to in new_temporal_orders:
                #if to.t0 not in self.validationTIDs or to.t1 not in self.validationTIDs:
                    #continue
                new_temporal_orders_[to.encode()] = to
            new_temporal_orders = list(new_temporal_orders_.values())
            # filter
            new_temporal_orders = self.filterTemporalOrders(new_temporal_orders)
            # 6. check if conflict and add more augmented training data
            data_aug_embedds, data_aug_processed, data_aug_tos = [], [], []
            start_time = time.time()
            if indicator == False:
                for to in new_temporal_orders:
                    embedd, t_processed, to_corr = self.creator.check(to, self.validation, self.dataProcessedMap, self.args.variants, DELIMITOR)
                    data_aug_embedds.append(embedd)
                    data_aug_tos.append(to_corr)
                    data_aug_processed.append(t_processed)
            else:
                data_aug_embedds, data_aug_processed, data_aug_tos = self.create_training_data_new(new_temporal_orders,
                                                                           deltaTOML, self.validation, self.dataProcessedMap)

            print('AUG : indicator = {}, the number of additional training data is {}'.format(indicator, len(data_aug_embedds)))
            print("6. The time of adding more training data is {}".format(time.time() - start_time))
            if len(data_aug_tos) == 0 or len(data_aug_embedds) == 0:
                break

            ''' if terminate
            '''
            if round >= MAX_ROUND:
                break
            acc_round = self.creator.evaluate(self.validationT)
            print("6.1 The accuracy of validationT is {}".format(acc_round))
            if acc_round <= accValidT:
                countTerminT += 1
            else:
                countTerminT = 0
            if countTerminT >= self.terminT:
                break
            accValidT = acc_round

            # 7. remove data_aug_tos from validation
            #self.addstableTOsKeyMap(stableTOsMap, data_aug_tos)
            #GS.addTOs(data_aug_tos)
            data_aug_embedds_, data_aug_processed_, data_aug_tos_ = [], [], []
            for oid, to in enumerate(data_aug_tos):
                ifConflict = GS.checkTOIsConflict(to)
                ifConflict_ = GS.checkIfConflictListTO(to)
                if ifConflict == False and ifConflict_ == False:
                    data_aug_embedds_.append(data_aug_embedds[oid])
                    data_aug_processed_.append(data_aug_processed[oid])
                    data_aug_tos_.append(to)
                    GS.addTO(to)
            data_aug_embedds, data_aug_processed, data_aug_tos = data_aug_embedds_, data_aug_processed_, data_aug_tos_
            # GS.addTOs(data_aug_tos)


            # improved_data, improved_data_processed = self.create_training_data(new_temporal_orders, conflicted_orders)
            logging.info('[LOG]: Training round ' + str(round) + ' finished ' + str(
                len(improved_data)) + ' new training instances found')

            overall_time_round = time.time() - start_overall_time_round
            overall_time += overall_time_round

            start_time = time.time()
            if self.args.variants == 'critic':
                self.critic.evaluationValid(self.validation, GS, round, self.ifInTrainMap)
                break
            elif self.args.variants == 'gate' or self.args.variants == 'gatenc' or self.args.variants == 'creatoritr':
                #self.creator.evaluationValid(self.validation, round, self.ifInTrainMap)
                #self.evaluationValid(self.validation, GS, round, self.ifInTrainMap)
                self.evaluationValid_new(self.validation, GS, round, self.ifInTrainMap)
            else:
                self.evaluationValid_new(self.validation, GS, round, self.ifInTrainMap)
            print("7. The time of evaluation is {}".format(time.time() - start_time))

            print('The overall time for {} round is {}'.format(round, overall_time_round))

            round += 1

        # print('The overall time for all rounds is {}'.format(overall_time))
        eval_start_time = time.time()
        logging.info('[LOG]: Training Finished, evaluation final performance ')
        if self.args.variants == 'critic':
            self.critic.evaluationValid(self.validation, GS, round, self.ifInTrainMap)
        elif self.args.variants == 'gate' or self.args.variants == 'gatenc':
            #self.creator.evaluationValid(self.validation, round, self.ifInTrainMap)
            #self.evaluationValid(self.validation, GS, round, self.ifInTrainMap)
            self.evaluationValid_new(self.validation, GS, round, self.ifInTrainMap)
        else:
            self.evaluationValid_new(self.validation, GS, round, self.ifInTrainMap)
        eval_time = time.time() - eval_start_time
        print('The overall time for all rounds is {} and the training time is {}'.format(overall_time + eval_time, eval_time))


    # only for ablation study
    def filterTemporalOrders(self, tos):
        tos_new = []
        for to in tos:
            if to.eid in self.usedEntitiesMap:
                tos_new.append(to)
        return tos_new

    def computeConfScore(self, tid, dist_, tids_detect, tidToScMap):
        maxScore, pos = -1e-10, -1
        dist_tid = dist_[tidToScMap[tid]];
        for i in range(len(tids_detect) + 1):
            # [_ 3 _ 5 _ 4 _]
            val = 0
            for z in range(0, i):
                tid_ = tids_detect[z]
                val += dist_tid - dist_[tidToScMap[tid_]]
            for z in range(i, len(tids_detect)):
                tid_ = tids_detect[z]
                val += dist_[tidToScMap[tid_]] - dist_tid
            if val >= maxScore:
                maxScore = val
                pos = i
        return maxScore, pos

    def computeConfScore_old(self, tid, dist_, tids_detect, tidsToScMap):
        dist_tid = dist_[tidsToScMap[tid]]
        for i, tid_ in enumerate(tids_detect):
            if dist_tid < dist_[tidsToScMap[tid_]]:
                return dist_tid, i
        return dist_tid, len(tids_detect)

    def computeValidScores(self, dists, _list, GS, attr, eid):
        ground_truth = [e[0] for e in _list]
        tidToScMap = defaultdict()
        for i, e in enumerate(_list):
            tidToScMap[e[0]] = i
        dists_ = [[e[0], d] for e, d in zip(_list, dists)]
        tids_detect = GS.sortGlobalOrder(attr, eid)
        # [3, 5]
        # [1, 2, 5, 3, 4]
        tids_remain = [tid for tid in ground_truth if tid not in tids_detect]
        np.random.seed(20)
        np.random.shuffle(tids_remain)
        while True:
            if len(tids_remain) <= 0:
                break
            maxScore, maxTid, maxPos = -1e-12, tids_remain[0], 0
            for tid in tids_remain:
                score, pos = self.computeConfScore(tid, dists, tids_detect, tidToScMap)
                if score >= maxScore:
                    maxScore = score
                    maxTid = tid
                    maxPos = pos
            # update
            tids_remain.remove(maxTid)
            # print(tids_remain)
            tids_detect = tids_detect[:maxPos] + [maxTid] + tids_detect[maxPos:]
        return tids_detect

    def evaluationValid(self, validation, GS, round, ifInTrainMap):
        ndcgmetric, mrr, TP, FP, FN, acc, count = 0, 0, 0, 0, 0, 0, 0
        TP_, FP_, FN_, acc_, count_ = 0, 0, 0, 0, 0
        if self.creator.model is None:
            from creator.GateCreator import Net
            model = Net(self.args.input_dim, self.args.embedding_dim)
            path = 'best_saved_weights.pt'
            model.load_state_dict(torch.load(path))
        validation_processed = validation['data_processed']
        for index, _list in enumerate(validation_processed):
            attr, eid = _list[0], _list[-1]
            attribute_emb = validation['data_attribute_embeddings'][attr]
            attribute_emb_new = torch.cat((attribute_emb, attribute_emb), 0)
            dist_timelienss = []
            for pair in _list[1]:
                context_index, attr_val = pair[0], pair[1]
                context_index_emb = validation['data_sentence_embeddings'][context_index]
                attr_val_emb = validation['data_attribute_embeddings'][str(attr_val)]
                if self.args.variants == 'creatornc':
                    context_index_emb = attr_val_emb
                val_emb = torch.cat((context_index_emb, attr_val_emb), 0)
                dist_ = self.creator.predictOneValue(attribute_emb_new, val_emb)
                dist_timelienss.append(dist_)
            prediction_order = self.computeValidScores(dist_timelienss, _list[1], GS, attr, eid)
            ground_truth_order = [e[0] for e in _list[1]]
            ndcg_s, mrr_s, TP_s, FP_s, FN_s, acc_s, count_s = metrics(prediction_order, ground_truth_order, attr, ifInTrainMap, index * 100)
            ndcgmetric += ndcg_s
            mrr += mrr_s
            TP += TP_s
            FP += FP_s
            FN += FN_s
            acc += acc_s
            count += count_s
            # FScore
            TP_ss, FP_ss, FN_ss, acc_ss, count_ss = FScore(dist_timelienss, ground_truth_order, attr, eid, GS, self.ifInTrainMap)
            TP_ += TP_ss
            FP_ += FP_ss
            FN_ += FN_ss
            acc_ += acc_ss
            count_ += count_ss

        ndcg = ndcgmetric * 1.0 / len(validation_processed)
        mrr = mrr * 1.0 / len(validation_processed)
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        Fmeasure = 2 * precision * recall / (precision + recall)
        accuracy = acc * 1.0 / count #len(validation_processed)

        logging.info("round={} ndcg={} mrr={} precision={} recall={} Fmeasure={}".format(round, ndcg, mrr, precision, recall, Fmeasure))
        print("roundGATE={} ndcg={} mrr={} precision={} recall={} Fmeasure={} accuracy={}".format(round, ndcg, mrr, precision, recall, Fmeasure, accuracy))

        precision = TP_ * 1.0 / (TP_ + FP_)
        recall = TP_ * 1.0 / (TP_ + FN_)
        Fmeasure = 2 * precision * recall / (precision + recall)
        accuracy = acc_ * 1.0 / count_ #len(validation_processed)
        print("roundFScore={} precision={} recall={} Fmeasure={} accuracy={}".format(round, precision, recall, Fmeasure, accuracy))

    ''' new version of evaluation by output results of both creator and gate
    '''
    def evaluationValid_new(self, validation, GS, round, ifInTrainMap):
        ndcgmetric, mrr, TP, FP, FN, acc, count = 0, 0, 0, 0, 0, 0, 0
        ndcgmetric_ml, mrr_ml, TP_ml, FP_ml, FN_ml, acc_ml, count_ml = 0, 0, 0, 0, 0, 0, 0
        if self.creator.model is None:
            from creator.GateCreator import Net
            model = Net(self.args.input_dim, self.args.embedding_dim)
            path = 'best_saved_weights.pt'
            model.load_state_dict(torch.load(path))
        validation_processed = validation['data_processed']
        
        # get all distances
        all_attrs_emb, all_vals_emb = [], []
        for index, _list in enumerate(validation_processed):
            attr, eid = _list[0], _list[-1]
            attribute_emb = validation['data_attribute_embeddings'][attr]
            attribute_emb_new = torch.cat((attribute_emb, attribute_emb), 0)
            for pair in _list[1]:
                context_index, attr_val = pair[0], pair[1]
                context_index_emb = validation['data_sentence_embeddings'][context_index]
                attr_val_emb = validation['data_attribute_embeddings'][str(attr_val)]
                if self.args.variants == 'creatornc':
                    context_index_emb = attr_val_emb
                val_emb = torch.cat((context_index_emb, attr_val_emb), 0)
                all_attrs_emb.append(attribute_emb_new)
                all_vals_emb.append(val_emb)
        all_attrs_emb = torch.stack(all_attrs_emb)
        all_vals_emb = torch.stack(all_vals_emb)
        all_dists, encode_attrs_emb, encode_vals_emb = self.creator.predictBatchValue(all_attrs_emb, all_vals_emb)
        start = 0
        
        for index, _list in enumerate(validation_processed):
            attr, eid = _list[0], _list[-1]
            dist_timelienss = all_dists[start: start + len(_list[1])]
            # print("Preprocess distances : ", all_dists[start: start + len(_list[1])])
            # print("Features 1 : ", all_attrs_emb.detach().cpu().numpy()[start: start + len(_list[1]), :5])
            # print("Features 2 : ", all_vals_emb.detach().cpu().numpy()[start: start + len(_list[1]), :5])
            start += len(_list[1])
            '''
            attribute_emb = validation['data_attribute_embeddings'][attr]
            attribute_emb_new = torch.cat((attribute_emb, attribute_emb), 0)
            dist_timelienss = []
            for pair in _list[1]:
                context_index, attr_val = pair[0], pair[1]
                context_index_emb = validation['data_sentence_embeddings'][context_index]
                attr_val_emb = validation['data_attribute_embeddings'][str(attr_val)]
                if self.args.variants == 'creatornc':
                    context_index_emb = attr_val_emb
                val_emb = torch.cat((context_index_emb, attr_val_emb), 0)
                dist_ = self.creator.predictOneValue(attribute_emb_new, val_emb)
                print(dist_, attribute_emb_new.detach().cpu().numpy()[:5], val_emb.detach().cpu().numpy()[:5])
                dist_timelienss.append(dist_)
            '''

            # print("Start prediction order ... ", index, _list)
            prediction_order = self.computeValidScores(dist_timelienss, _list[1], GS, attr, eid)
            # print("After prediction order ... ", prediction_order)
            ground_truth_order = [e[0] for e in _list[1]]

            prediction_order_ml = [[tid, dist_] for tid, dist_ in zip(ground_truth_order, dist_timelienss)]
            np.random.seed(90)
            np.random.shuffle(prediction_order_ml)
            prediction_order_ml = sorted(prediction_order_ml, key=lambda x : x[1])
            prediction_order_ml = [e[0] for e in prediction_order_ml]
            ground_truth_order_ml = copy.deepcopy(ground_truth_order)

            ndcg_s, mrr_s, TP_s, FP_s, FN_s, acc_s, count_s = metrics(prediction_order, ground_truth_order, attr, ifInTrainMap, index * 100)
            ndcgmetric += ndcg_s
            mrr += mrr_s
            TP += TP_s
            FP += FP_s
            FN += FN_s
            acc += acc_s
            count += count_s

            # Creator !!!
            #TP_ss, FP_ss, FN_ss, acc_ss, count_ss = FScore(dist_timelienss, ground_truth_order, attr, eid, GS, self.ifInTrainMap)
            ndcg_ss, mrr_ss, TP_ss, FP_ss, FN_ss, acc_ss, count_ss = metrics(prediction_order_ml, ground_truth_order_ml, attr, ifInTrainMap, index * 100)
            ndcgmetric_ml += ndcg_ss
            mrr_ml += mrr_ss
            TP_ml += TP_ss
            FP_ml += FP_ss
            FN_ml += FN_ss
            acc_ml += acc_ss
            count_ml += count_ss

        ndcg = ndcgmetric * 1.0 / len(validation_processed)
        mrr = mrr * 1.0 / len(validation_processed)
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        Fmeasure = 2 * precision * recall / (precision + recall)
        accuracy = acc * 1.0 / count #len(validation_processed)

        logging.info("round={} ndcg={} mrr={} precision={} recall={} Fmeasure={}".format(round, ndcg, mrr, precision, recall, Fmeasure))
        print("roundGATE={} ndcg={} mrr={} precision={} recall={} Fmeasure={} accuracy={}".format(round, ndcg, mrr, precision, recall, Fmeasure, accuracy))

        ndcg = ndcgmetric_ml * 1.0 / len(validation_processed)
        mrr = mrr_ml * 1.0 / len(validation_processed)
        precision = TP_ml * 1.0 / (TP_ml + FP_ml)
        recall = TP_ml * 1.0 / (TP_ml + FN_ml)
        Fmeasure = 2 * precision * recall / (precision + recall)
        accuracy = acc_ml * 1.0 / count_ml #len(validation_processed)
        #print("roundFScore={} precision={} recall={} Fmeasure={} accuracy={}".format(round, precision, recall, Fmeasure, accuracy))
        print("roundCreator={} ndcg={} mrr={} precision={} recall={} Fmeasure={} accuracy={}".format(round, ndcg, mrr, precision, recall, Fmeasure, accuracy))



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

    def choose_high_confidence_(self, validation, high_conf_sample_ratio, confThreshold, stableTOsMap, maxAdditionalAugData):

        def toEncode(t0, t1, attr, eid):
            return str(t0) + ETID_ATTR_DELIMITOR + str(t1) + ETID_ATTR_DELIMITOR + str(attr) + '_' + str(eid)

        logging.info("Choosing high confidence temporal orders")
        if self.creator.model is None:
            from creator.GateCreator import Net
            model = Net(self.args.input_dim, self.args.embedding_dim)
            path = 'best_saved_weights.pt'
            model.load_state_dict(torch.load(path))


        
        # Notice that validation_processed does not have orders!
        # Faster version
        validation_processed = validation['data_processed']
        res_data_processed = []
        tmp_batch_attr_embs, tmp_batch_val_embs, tmp_batch_pairs = [], [], []
        for index, _list in random.sample(list(enumerate(validation_processed)), int(len(validation_processed) * high_conf_sample_ratio)):
            attr, eid = _list[0], _list[-1]
            attribute_emb = validation['data_attribute_embeddings'][attr]
            for ele in _list[1]:
                context_index, attr_val = ele[0], ele[1]
                # get embeddings
                context_index_emb = validation['data_sentence_embeddings'][context_index]
                attr_val_emb = validation['data_attribute_embeddings'][str(attr_val)]
                if self.args.variants == 'creatornc':
                    context_index_emb = attr_val_emb
                # concatenate all of the tensors
                attribute_emb_new = torch.cat((attribute_emb, attribute_emb), 0)
                val_emb = torch.cat((context_index_emb, attr_val_emb), 0)
                tmp_batch_attr_embs.append(attribute_emb_new)
                tmp_batch_val_embs.append(val_emb)
            tmp_batch_pairs.append(_list)
        tmp_batch_attr_embs = torch.stack(tmp_batch_attr_embs)
        tmp_batch_val_embs = torch.stack(tmp_batch_val_embs)
        print('tmp batch', tmp_batch_attr_embs.shape, tmp_batch_val_embs.shape)
        # ML inference
        dist_, _, _ = self.creator.predictBatchValue(tmp_batch_attr_embs, tmp_batch_val_embs)
        start = 0
        for index, _list in enumerate(tmp_batch_pairs):
            attr, eid = _list[0], _list[-1]
            dist_one = dist_[start: start + len(_list[1])]
            '''
            dist_one_sc = [[sc, d] for sc, d in zip(_list[1], dist_one)]
            dist_one_sc = sorted(dist_one_sc, key=lambda x : x[-1])
            for i in range(1, len(dist_one_sc), 1):
                dist_pos_sc, dist_neg_sc = dist_one_sc[i-1], dist_one_sc[i]
                diff = dist_neg_sc[1] - dist_pos_sc[1]
                diff = 1.0 / (1 +np.exp(-diff))
                if diff >= confThreshold:
                    res_data_processed.append([attr, dist_pos_sc[0], dist_neg_sc[0], eid, diff])
            '''
            script = np.arange(len(_list[1]))
            for pair_sc in itertools.permutations(script, 2):
                sc_1, sc_2 = pair_sc
                pair = [_list[1][sc_1], _list[1][sc_2]]
                dist_pos, dist_neg = dist_one[sc_1], dist_one[sc_2]
                diff = dist_neg - dist_pos
                diff = 1.0 / (1 + np.exp(-(diff) ))
                kk_1, kk_2 = toEncode(pair[0], pair[1], attr, eid), toEncode(pair[1], pair[0], attr, eid)
                if diff >= confThreshold and kk_1 not in stableTOsMap and kk_2 not in stableTOsMap:
                    res_data_processed.append([attr, pair[0], pair[1], eid, diff])
            
            start += len(_list[1])


        res_data_processed = sorted(res_data_processed, key=lambda x: x[-1], reverse=False)
        res_data_processed = [e[:-1] for e in res_data_processed[:maxAdditionalAugData]]
        # transform data_processed to TOs
        tos = self.transformTOs(res_data_processed)
        return tos
        

        ''' #Old version 
        # Notice that validation_processed does not have orders!
        validation_processed = validation['data_processed']
        res_data_processed = []
        for index, _list in random.sample(list(enumerate(validation_processed)), int(len(validation_processed) * high_conf_sample_ratio)):
            attr, eid = _list[0], _list[-1]
            attribute_emb = validation['data_attribute_embeddings'][attr]
            for pair in itertools.combinations(_list[1], 2):
                context_index_1, attr_val_1 = pair[0][0], pair[0][1]
                context_index_2, attr_val_2 = pair[1][0], pair[1][1]
                # check whether the temporal orders have already been in the stable set
                if self.stableTOKey(context_index_1, context_index_2, attr, eid) in stableTOsMap or self.stableTOKey(context_index_2, context_index_1, attr, eid) in stableTOsMap:
                    continue
                # get embeddings
                context_index_emb_1 = validation['data_sentence_embeddings'][context_index_1]
                attr_val_emb_1 = validation['data_attribute_embeddings'][str(attr_val_1)]
                context_index_emb_2 = validation['data_sentence_embeddings'][context_index_2]
                attr_val_emb_2 = validation['data_attribute_embeddings'][str(attr_val_2)]
                if self.args.variants == 'creatornc':
                    context_index_emb_1 = attr_val_emb_1
                    context_index_emb_2 = attr_val_emb_2
                # concate all of the tensors
                attribute_emb_new = torch.cat((attribute_emb, attribute_emb), 0)
                val_emb_1 = torch.cat((context_index_emb_1, attr_val_emb_1), 0)
                val_emb_2 = torch.cat((context_index_emb_2, attr_val_emb_2), 0)
                diff, flagC = self.creator.predictHighConf([attribute_emb_new, val_emb_1, val_emb_2], confThreshold)
                if flagC:
                    res_data_processed.append([attr, pair[0], pair[1], eid, diff])

        res_data_processed = sorted(res_data_processed, key=lambda x : x[-1], reverse=False)
        res_data_processed = [e[:-1] for e in res_data_processed[:maxAdditionalAugData]]
        # transform data_processed to TOs
        tos = self.transformTOs(res_data_processed)
        return tos
        '''        

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
            for index, i in random.sample(list(enumerate(training_processed)), int(len(training_processed) * high_conf_sample_ratio)):
                attribute = i[0]
                pos_context_index, pos_att = i[1]
                neg_context_index, neg_att = i[2]

                attribute_emb = self.training['data_attribute_embeddings'][attribute]
                pos_context_emb = self.training['data_sentence_embeddings'][pos_context_index]
                pos_att_emb = self.training['data_attribute_embeddings'][str(pos_att)]
                neg_context_emb = self.training['data_sentence_embeddings'][neg_context_index]
                neg_att_emb = self.training['data_attribute_embeddings'][str(neg_att)]
                if self.args.variants == 'creatornc':
                    pos_context_emb = pos_att_emb
                    neg_context_emb = neg_att_emb

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

    def create_one_training_data_to(self, to, validation, dataProcessedMap):
        '''
        sc = dataProcessedMap[to.attribute + DELIMITOR + str(to.t0)]
        neg = self.validation['data_processed'][sc[0]][1][sc[1]]
        sc = dataProcessedMap[to.attribute + DELIMITOR + str(to.t1)]
        pos = self.validation['data_processed'][sc[0]][1][sc[1]]
        '''
        neg = [to.t0, self.data[to.t0][self.attrsMap[to.attribute]]]
        pos = [to.t1, self.data[to.t1][self.attrsMap[to.attribute]]]


        # get embeddings
        attribute_emb = validation['data_attribute_embeddings'][to.attribute]
        context_index_emb_neg = validation['data_sentence_embeddings'][neg[0]]
        attr_val_emb_neg = validation['data_attribute_embeddings'][str(neg[1])]
        context_index_emb_pos = validation['data_sentence_embeddings'][pos[0]]
        attr_val_emb_pos = validation['data_attribute_embeddings'][str(pos[1])]
        if self.args.variants == 'creatornc':
            context_index_emb_pos = attr_val_emb_pos
            context_index_emb_neg = attr_val_emb_neg

        # concat all of the tensors
        attribute_emb = torch.cat((attribute_emb, attribute_emb), 0)
        pos_instance = torch.cat((context_index_emb_pos, attr_val_emb_pos), 0)
        neg_instance = torch.cat((context_index_emb_neg, attr_val_emb_neg), 0)
        return attribute_emb, pos_instance, neg_instance, to.attribute, pos, neg

    def create_training_data_new(self, new_temporal_orders, deltaTOsML, validation, dataProcessedMap):

        data_aug_embedds, data_aug_processed, data_aug_tos = [], [], []
        for to in new_temporal_orders:
            attribute_emb, pos_instance, neg_instance, attribute, pos, neg = self.create_one_training_data_to(to, validation, dataProcessedMap)
            data_aug_embedds.append([attribute_emb, pos_instance, neg_instance])
            data_aug_processed.append([attribute, pos, neg, to.eid])

        for to in deltaTOsML:
            attribute_emb, pos_instance, neg_instance, attribute, pos, neg = self.create_one_training_data_to(to, validation, dataProcessedMap)
            data_aug_embedds.append([attribute_emb, pos_instance, neg_instance])
            data_aug_processed.append([attribute, pos, neg, to.eid])

        data_aug_tos = self.transformTOs(data_aug_processed)

        return data_aug_embedds, data_aug_processed, data_aug_tos

    '''
    def create_training_data(self, new_temporal_orders, conflicted_orders):
        data = self.get_embeddings(new_temporal_orders)
        data += self.get_embeddings(conflicted_orders)
        return data, {**new_temporal_orders, **conflicted_orders}
    '''

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
