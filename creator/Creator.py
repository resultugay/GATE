from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

class Creator(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def prepare_data(self,remove_list):
        pass

    def save_vectors(self,col,vector):
        f = open('output_vectors/' + col + ".txt", "w")
        for key, value in vector.items():
            f.write(str(key) + '\t')
            for elem in value:
                f.write("%s\t" % elem.item())
            f.write('\n')

        f.close()

    def save_img(self,col,vector):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        vec_stat = vector[col]
        embedding_vectors = []
        x_s = []
        y_s = []
        for key, value in vector.items():
            #embedding_vectors.append(value.detach().numpy())

            x_s.append(value.detach().numpy().tolist()[0])
            y_s.append(value.detach().numpy().tolist()[1])

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(x_s, y_s, c='white')

        for idx, word in sorted(self.index_word[col].items()):
            x_coord = x_s[idx]
            y_coord = y_s[idx]
            ax.annotate(
                word,
                (x_coord, y_coord),
                horizontalalignment='center',
                verticalalignment='center',
                size=20,
                alpha=0.7
            )
            ax.set_title(f"Column-{col}")
        plt.savefig(f"Column-{col}.jpg")

    def create_currency_constraints(self,col,vector,cc):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        vec_stat = vector[col]
        latest = []
        non_latest = []
        for key, value in vector.items():
            res = cos(vec_stat, value)
            # print(key,res.item())
            if key != col:
                if res.item() > 0.5:
                    latest.append(key)
                else:
                    non_latest.append(key)

        for l in latest:
            for nl in non_latest:
                if col not in cc:
                    cc[col] = []
                cc[col].append((l,nl))
