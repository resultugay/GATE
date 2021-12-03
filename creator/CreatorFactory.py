from .LTRCreator import LTRCreator
from .MultiLabelCreator import MultiLabelCreator
from .A2VCreator import A2VCreator
import logging

class CreatorFactory:

    def __init__(self):
        pass

    def get_creator(args):
        if args.creator == 'LTR':
            return LTRCreator(args)
        elif args.creator == 'Multi':
            return MultiLabelCreator(args)
        elif args.creator == 'A2V':
            return A2VCreator(args)
        else:
            logging.error('No such creator name as ' + args.creator)
            raise ValueError(args.creator)
