from .DenialConstraintCritic import DenialConstraintCritic

class CriticFactory:
    def get_critic(name):
        if name == 'Denial':
            return DenialConstraintCritic()
        else:
            raise ValueError(name)
