import torch


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


class GateValidationTestDataset():
    def __init__(self, valid_or_test):
        self.data_processed = valid_or_test['data_processed']
        self.data_sentence_emb = valid_or_test['data_sentence_embeddings']
        self.data_attribute_emb = valid_or_test['data_attribute_embeddings']

        self.data = []
        for attribute, values in self.data_processed:
            attribute_emb = self.data_attribute_emb[attribute]
            attribute_emb = torch.cat((attribute_emb, attribute_emb), 0)
            order_emb = [attribute_emb]
            for context_id, val in values:
                context_emb = self.data_sentence_emb[context_id]
                val_emb = self.data_attribute_emb[str(val)]
                val_emb = torch.cat((context_emb, val_emb), 0)
                order_emb.append(val_emb)
            self.data.append(order_emb)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_processed)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data[index]
