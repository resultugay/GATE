from abc import ABC, abstractmethod

class Critic(ABC):

    @abstractmethod
    def choose_rules(self,currency_constraints,training_data):
        pass