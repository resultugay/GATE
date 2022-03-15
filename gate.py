import logging
from creator.CreatorFactory import CreatorFactory
import torch

class Gate:
    _instance = None
    training_data = None

    def __new__(cls):
        if cls._instance is None:
            logging.info('Creating the GATE object')
            cls._instance = super(Gate, cls).__new__(cls)
        else:
            logging.info('GATE object already exist')
        return cls._instance

    def initialize(self, args):
        logging.info('GATE initializing')
        logging.info('Data loading')
        self.training_data = torch.load(args.training+'training_embedded.pt')
        logging.info('Data loaded')
        self.creator = CreatorFactory.get_creator(args, self.training_data)
        #self.critic = CriticFactory.get_critic(args)
        self.args = args
        logging.info('GATE initialized')


    def train(self):
        logging.info('Training Started')
        self.creator.train()
        # self.evaluate()
        # Here we create the EM algorithm
        logging.info('Training Finished')
