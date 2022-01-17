import pandas as pd

from creator.CreatorFactory import CreatorFactory
from critic.CriticFactory import CriticFactory
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import logging
import pickle


class Gate:
    _instance = None
    creator = None
    critic = None
    data_training = None
    data_test = None
    args = None

    def __new__(cls):
        if cls._instance is None:
            logging.info('Creating the GATE object')
            cls._instance = super(Gate, cls).__new__(cls)
        else:
            logging.info('GATE object alread exist')
            return cls._instance
        return cls._instance

    def initialize(self, args):
        logging.info('GATE initializing')
        self.data_training = pd.read_csv(args.training_data)
        self.data_training = self.data_training.astype(str)
        self.data_test = pd.read_csv(args.test_data)
        #self.data_test = self.data_test.astype(str)
        self.creator = CreatorFactory.get_creator(args, self.data_training)
        self.critic = CriticFactory.get_critic(args)
        self.args = args
        logging.info('GATE initialized')

    def e_step(self):
        currency_constraints = self.creator.train()
        remove_list = self.critic.choose_rules(currency_constraints)
        return remove_list

    def m_step(self, remove_list):
        self.creator.prepare_data(remove_list)

    def train(self):
        if self.args.training:
            logging.info('Training Started')
            for i in range(2):
                remove_list = self.e_step()
                # self.evaluate()
                self.m_step(remove_list)
            # self.evaluate()
            # Here we create the EM algorithm
            logging.info('Training Finished')
        else:
            logging.info('No Training')

    def evaluate(self):
        if self.args.test:
            logging.info('Evaluation Started')
            self.data_test.index = self.data_test.id
            with open('cc.pickle', 'rb') as handle:
                cc = pickle.load(handle)

            vectors, columns = self.read_and_load_word_vectors()
            tp, fp = self.calculate_metrics_for_temporal_cols(columns, vectors)
            tp, fn = self.calculate_metrics_for_non_temporal_cols(self.args.non_temporal_columns, cc, tp)
            self.calculate_metrics(tp, 0, fp, fn)
            logging.info('Evaluation Finished')
        else:
            logging.info('No Test')

    def calculate_metrics(self,tp, tn, fp, fn):
        accuracy = (tp + tn) / (tp + fn + tn + fp)
        recall = tp / (fn + tp)
        precision = tp / (fp + tp)
        f_measure = (2 * precision * recall) / (precision + recall)
        logging.info('Data is :' + self.args.test_data)
        logging.info('ACCURACY :'+ str(accuracy))
        logging.info('RECALL :'+ str(recall))
        logging.info('PRECISION :'+ str(precision))
        logging.info('F-MEASURE :' + str(f_measure))

    def calculate_metrics_for_non_temporal_cols(self, non_temp_cols, cc, tp):
        fn = 0
        for i in self.data_test.id.unique():
            sub_df = self.data_test.loc[i]
            for col in non_temp_cols:
                max_timestamp = sub_df['timestamp'].max()
                sub_df.index = sub_df.row_id
                res = set()
                for ccs in cc[col]:
                    for line in ccs:
                        cc_col = line[0]
                        value = line[1]
                        for row_id in np.where(sub_df[cc_col] == value)[0]:
                            res.add(sub_df.loc[row_id][col])
                            if len(res) > 1:
                                break
                        if len(res) > 1:
                            break
                    if len(res) > 1:
                        fn += 1
                        break
                    elif len(res) != 0:
                        value = res.pop()
                        result_list = list(sub_df.loc[sub_df[col] == value]['timestamp'])
                        if max_timestamp in result_list:
                            tp += 1
                            break
                    else:
                        fn += 1
                        break

        return tp, fn

    def calculate_metrics_for_temporal_cols(self, columns, vectors):
        tp = 0
        fp = 0
        for i in self.data_test.id.unique():
            sub_df = self.data_test.loc[i]
            for col in columns:
                values_and_timestamps = sub_df[[col, 'timestamp']].to_dict('split')['data']
                a = vectors[col].get(col)
                max_score = -1
                attribute_value = ''
                max_time_stamp_of_value = 0
                max_timestamp = sub_df['timestamp'].max()
                min_timestamp = sub_df['timestamp'].min()
                for value, timestamp in values_and_timestamps:
                    b = vectors[col].get(str(value))
                    cos_sim = dot(a, b) / (norm(a) * norm(b))
                    if cos_sim > max_score:
                        max_score = cos_sim
                        attribute_value = value
                        max_time_stamp_of_value = timestamp
                    # print(value,cos_sim)
                # print(sub_df.index[0],'value for', col, ' is', attribute_value)
                # print(max_timestamp,timestamp)
                if max_timestamp == max_time_stamp_of_value:
                    tp += 1
                else:
                    fp += 1
                    # print(sub_df.index[0],'value for', col, ' is', attribute_value, max_timestamp,max_time_stamp_of_value)
        return tp, fp

    def load_vectors(self,col):
        try:
            vector = {}
            path = 'output_vectors/' + str(col) + '.txt'
            with open(path, 'r') as f:
                for line in f:
                    split_line = line.split('\t')
                    word = split_line[0]
                    embedding = np.array(split_line[1:-1], dtype=np.float64)
                    vector[word] = embedding
            logging.info(str(col) + " vector loaded")

            return vector
        except:
            logging.info('could not load vector for ' + str(col))
            return None

    def read_and_load_word_vectors(self):
        vectors = {}
        columns = []
        for col in self.data_test.columns:
            vector = self.load_vectors(col)
            if vector is not None:
                vectors[col] = vector
                columns.append(col)
        return vectors, columns
