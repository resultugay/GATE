import torch


class AV2Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Load data and get label
        X = self.list_IDs[index]
        y = self.labels[index]

        return X, y
