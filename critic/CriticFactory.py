from .DenialConstraintCritic import DenialConstraintCritic
from .LogicRuleCritic import LogicRuleCritic
from .MeasuresCritic import MeasuresCritic
import logging



class CriticFactory:
    def get_critic(args):
        if args.critic == 'Denial':
            return DenialConstraintCritic(args)
        elif args.critic == 'Logic':
            return LogicRuleCritic(args)
        elif args.critic == 'Measures':
            return MeasuresCritic(args)
        else:
            logging.error('No such critic name as ' + args.critic)
            raise ValueError(args.critic)
