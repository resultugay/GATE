from abc import ABC, abstractmethod

class Critic(ABC):

    @abstractmethod
    def read_CC(self):
        pass

    @abstractmethod
    def choose_rules(self,currency_constraints):
        pass