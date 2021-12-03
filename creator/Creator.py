from abc import ABC, abstractmethod

class Creator(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def prepare_data(self):
        pass