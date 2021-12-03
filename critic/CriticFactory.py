from .DenialConstraintCritic import DenialConstraintCritic
import logging

class CriticFactory:
    def get_critic(args):
        if args.critic == 'Denial':
            return DenialConstraintCritic(args.dataset)
        else:
            logging.error('No such critic name as ' + args.critic)
            raise ValueError(args.critic)
