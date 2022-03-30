from abc import ABC, abstractmethod


class Creator(ABC):

    @abstractmethod
    def train(self,training_data):
        pass
