import logging
from .Critic import Critic


class DenialConstraintCritic(Critic):
    def __init__(self,dataset):
        logging.info('Critic is Denial')
        pass

    def train(self):
        pass
