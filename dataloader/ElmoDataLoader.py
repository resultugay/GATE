import torch
import tensorflow_hub as hub
import tensorflow as tf

class ElmoDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, elmo, index_word):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.elmo_vectors = elmo
        self.index_word = index_word
        self.embeddings = {}
        for attribute in index_word.values():
            embeds = elmo.signatures["default"](tf.constant([str(attribute)]))["elmo"]
            embeds = torch.tensor(embeds.numpy())
            embeds = embeds.reshape(embeds.shape[1], -1)
            embeds = embeds.mean(dim=0)
            self.embeddings[attribute] = embeds

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Load data and get label
        X = []
        ids_ = self.list_IDs[index]
        for i in ids_:
            word = self.index_word[i.item()]
            X.append(self.embeddings[word])

        y = self.labels[index]

        return X, y
