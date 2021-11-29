from .DenialConstraintCritic import DenialConstraintCritic
import logging

class CriticFactory:
    def get_critic(name,dataset):
        if name == 'Denial':
            return DenialConstraintCritic(dataset)
        else:
            logging.error('No such critic name as ' + name)
            raise ValueError(name)
