import logging
from .Creator import Creator

class LTRCreator(Creator):

    def __init__(self,dataset):
        logging.info('Creator is LTR')
        pass

    def train(self):
        pass

    def evaluate(self):
        pass
