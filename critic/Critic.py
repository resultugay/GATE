from abc import ABC, abstractmethod

class Critic(ABC):

    @abstractmethod
    def train(self):
        pass
