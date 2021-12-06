from abc import ABC, abstractmethod

class Creator(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def prepare_data(self):
        pass

    def save_vectors(self,col,vector):
        f = open('output_vectors/' + col + "_vectors.txt", "w")
        for key, value in vector.items():
            f.write(str(key) + ' ')
            for elem in value:
                f.write("%s " % elem.item())
            f.write('\n')

        f.close()