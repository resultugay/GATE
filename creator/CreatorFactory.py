from abc import ABCMeta, abstractmethod
from .LtrCreator import LtrCreator

class CreatorFactory:

    def __init__(self):
        pass

    def get_creator(name):
        if name == 'LTR':
            return LtrCreator()
        elif name == 'MultiLabel':
            return MultiLabelCreator()
        else:
            raise ValueError(name)
