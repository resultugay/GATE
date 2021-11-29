from .LTRCreator import LTRCreator
from .MultiLabelCreator import MultiLabelCreator
import logging

class CreatorFactory:

    def __init__(self):
        pass

    def get_creator(name,dataset):
        if name == 'LTR':
            return LTRCreator(dataset)
        elif name == 'MultiLabel':
            return MultiLabelCreator(dataset)
        else:
            logging.error('No such creator name as ' + name)
            raise ValueError(name)
