import logging
from .Creator import Creator

class MultiLabelCreator(Creator):

    def __init__(self,dataset):
        logging.info('Creator is MultiLabel')

    def train(self):
        pass

    def evaluate(self):
        pass
