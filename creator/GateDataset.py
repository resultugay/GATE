import torch


class GateDataset():
    def __init__(self, improved_data, training_data):
        if len(improved_data) > 2:
            self.training_instances = improved_data
        else:
            self.data_processed = training_data['data_processed']
            self.attribute_embeddings = training_data['data_attribute_embeddings']
            self.sentence_embeddings = training_data['data_sentence_embeddings']
            self.training_instances = []

    def __len__(self):
        'Denotes the total number of samples'
        if self.training_instances:
            return len(self.training_instances)
        else:
            return len(self.data_processed)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if self.training_instances:
            data = self.training_instances[index]
            return data

        attribute, (pos_context_index, pos_att), (neg_context_index, neg_att) = self.data_processed[index]
        attribute_emb = self.attribute_embeddings[attribute]
        pos_context_emb = self.sentence_embeddings[pos_context_index]
        pos_att_emb = self.attribute_embeddings[str(pos_att)]
        neg_context_emb = self.sentence_embeddings[neg_context_index]
        neg_att_emb = self.attribute_embeddings[str(neg_att)]

        # concat all of the tensors
        attribute_emb = torch.cat((attribute_emb, attribute_emb), 0)
        pos_instance = torch.cat((pos_context_emb, pos_att_emb), 0)
        neg_instance = torch.cat((neg_context_emb, neg_att_emb), 0)

        return attribute_emb, pos_instance, neg_instance


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
