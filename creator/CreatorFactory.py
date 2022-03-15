import logging

from .GateCreator import GateCreator


class CreatorFactory:

    def __init__(self):
        pass

    def get_creator(args, training_data):
        if args.creator == 'Gate':
            return GateCreator(args, training_data)
        else:
            logging.error('No such creator name as ' + args.creator)
            raise ValueError('Define the Creator')
