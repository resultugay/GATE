from abc import ABC, abstractmethod

class Creator(ABC):

    @abstractmethod
    def train(self):
        pass
