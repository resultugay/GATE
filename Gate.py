from creator.CreatorFactory import CreatorFactory
from critic.CriticFactory import CriticFactory
import logging

class Gate:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the object')
            cls._instance = super(Gate, cls).__new__(cls)
        return cls._instance


    def initialize(self,args):
        logging.info('GATE initializing')
        self.creator = CreatorFactory.get_creator(args.creator)
        self.critic = CriticFactory.get_critic(args.critic)
        logging.info('GATE initialized')


    def train(self):
        logging.info('Training Started')
        #Here we create the EM algorithm
        logging.info('Training Finished')

    def evaluate(self):
        logging.info('Evaluation Started')
        #Here we create the EM algorithm
        logging.info('Evaluation Finished')
