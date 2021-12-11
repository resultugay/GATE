from .DenialConstraintCritic import DenialConstraintCritic
from .LogicRuleCritic import LogicRuleCritic
import logging

class CriticFactory:
    def get_critic(args):
        if args.critic == 'Denial':
            return DenialConstraintCritic(args)
        elif args.critic == 'Logic':
            return LogicRuleCritic(args)
        else:
            logging.error('No such critic name as ' + args.critic)
            raise ValueError(args.critic)
