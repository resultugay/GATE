import pandas as pd

from creator.CreatorFactory import CreatorFactory
from critic.CriticFactory import CriticFactory
import logging


class Gate:
    _instance = None
    creator = None
    critic = None
    data_training = None

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
        self.data_training = pd.read_csv(args.dataset)
        self.data_training = self.data_training.astype(str)
        self.creator = CreatorFactory.get_creator(args, self.data_training)
        self.critic = CriticFactory.get_critic(args)
        logging.info('GATE initialized')

    def e_step(self):
        currency_constraints = self.creator.train()
        remove_list = self.critic.choose_rules(currency_constraints)
        return remove_list

    def m_step(self, remove_list):
        self.creator.prepare_data(remove_list)

    def train(self):
        logging.info('Training Started')
        for i in range(2):
            remove_list = self.e_step()
            # self.evaluate()
            self.m_step(remove_list)
        # self.evaluate()
        # Here we create the EM algorithm
        logging.info('Training Finished')

    def evaluate(self):
        logging.info('Evaluation Started')
        # Here we create the EM algorithm
        logging.info('Evaluation Finished')
