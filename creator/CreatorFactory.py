from .LTRCreator import LTRCreator
from .MultiLabelCreator import MultiLabelCreator
from .A2VCreator import A2VCreator
import logging

class CreatorFactory:

    def __init__(self):
        pass

    def get_creator(args,data_training):
        if args.creator == 'LTR':
            return LTRCreator(args,data_training)
        elif args.creator == 'Multi':
            return MultiLabelCreator(args,data_training)
        elif args.creator == 'A2V':
            logging.info('Creator is A2V')
            return A2VCreator(args,data_training)
        else:
            logging.error('No such creator name as ' + args.creator)
            raise ValueError(args.creator)
