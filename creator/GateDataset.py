class GateDataset():
    def __init__(self, training_instances):
        self.training_instances = training_instances

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.training_instances)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data = self.training_instances[index]
        return data
