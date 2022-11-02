import numpy as np
import pandas as pd
import torch
import transformers as ppb
import pickle
import os
import argparse
import itertools
from collections import defaultdict
from sklearn.model_selection import train_test_split

meaninglessCols = ['id', 'entity_id', 'row_id', 'timestamp']

def main():
    parser = argparse.ArgumentParser(description="Pre-process data")

    parser.add_argument('-data_dir', '--data_dir', type=str, default='../../data/nba')
    parser.add_argument('-filename', '--filename', type=str, default='data')
    parser.add_argument('-timelinessAttrs', '--timelinessAttrs', type=str, default="team_abbreviation, player_weight, college")
    parser.add_argument('-training_ratio', '--training_ratio', type=float, default=0.1)
    parser.add_argument('--testing_ratio', '--testing_ratio', type=float, default=0.4)

    args = parser.parse_args()
    arg_dict = args.__dict__

    # load model
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    print('Finish Loading Bert ............')

    def serialize(row, cols):
        context = ''
        for c in cols:
            if c not in meaninglessCols:
                context += '[COL] ' + str(c) + ' [VAL] ' + str(row[c])

        token_ids = tokenizer.encode(context, add_special_tokens=True)
        return token_ids

    data = pd.read_csv(os.path.join(arg_dict['data_dir'], arg_dict['filename'] + '.csv'))
    data['serialized'] = data.apply(serialize, axis=1, args=([data.columns]))
    data_dict = data[['id', 'serialized']].to_dict('list')
    context = dict(zip(data_dict['id'], data_dict['serialized']))
    data = data.iloc[:, 0:len(data.columns) - 1]
    max_len = 0
    for i in context.values():
        if len(i) > max_len:
            max_len = len(i)
    context = dict([(key, (values + [0] * (max_len - len(values)))) for key, values in context.items()])
    # truncate 512
    for key in context.keys():
        context[key] = context[key][:512]

    print('Finish Generating the context information ............')

    attention_mask = np.where(np.array(list(context.values())) != 0, 1, 0)
    sentence_embeddings = []
    input_ids = torch.tensor(list(context.values()))
    batch_size = 1024
    for i in range(len(input_ids) // batch_size + 1):
        batch_start, batch_end = i * batch_size, min((i + 1) * batch_size, len(input_ids))
        if batch_end <= batch_start:
            break
        attention_mask_s = torch.tensor(attention_mask[batch_start: batch_end])
        input_ids_s = input_ids[batch_start: batch_end]
        with torch.no_grad():
            model_output = model(input_ids_s, attention_mask=attention_mask_s)

        '''
        tokenized = {}
        for index, key in enumerate(context.keys()):
            average = torch.mean(model_output[0][index], 0, True)
            concatted_rep = torch.cat((average.flatten(), model_output[0][index][0]))
            tokenized[key] = concatted_rep
        '''

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        print('Generating Context Embeddings ............')
        # 1. context embeddings for all tuples in D
        sentence_embeddings_s = mean_pooling(model_output, attention_mask_s)
        print("The shape of sentence embeddings is ", sentence_embeddings_s.shape)
        print("The first 10 sentence embeddings are ", sentence_embeddings_s[:10])
        if i == 0:
            sentence_embeddings = sentence_embeddings_s
        else:
            sentence_embeddings = torch.cat((sentence_embeddings, sentence_embeddings_s), 0)

    torch.save(sentence_embeddings, os.path.join(arg_dict['data_dir'], arg_dict['filename'] + '_sentence_embeddings.pt'))

    print("Generate attribute embeddings ............")
    columns = [x for x in list(data.columns) if x not in meaninglessCols + ['serialized']]
    all_attribute_values = set()
    for key, value in data[columns].to_dict('list').items():
        all_attribute_values.update(set(value))
        all_attribute_values.add(key)
    all_attribute_values = [str(i) for i in all_attribute_values]
    attribute_values_encoded = {}
    for att_val in all_attribute_values:
        attribute_values_encoded[att_val] = tokenizer.encode(att_val, add_special_tokens=False)

    max_len = 0
    for i in attribute_values_encoded.values():
        if len(i) > max_len:
            max_len = len(i)
    context = dict(
        [(key, (values + [0] * (max_len - len(values)))) for key, values in attribute_values_encoded.items()])
    # trunctate 512
    for key in context.keys():
        context[key] = context[key][:512]

    attention_mask = np.where(np.array(list(context.values())) != 0, 1, 0)
    input_ids = torch.tensor(list(context.values()))
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        model_output = model(input_ids, attention_mask=attention_mask)

    # 2. attribute embeddings for D
    attribute_embeddings = model_output['last_hidden_state'][:, 0, :]
    attribute_embeddings = dict(zip(context.keys(), attribute_embeddings.detach().cpu().numpy()))
    attribute_embeddings = {key: torch.tensor(value) for key, value in attribute_embeddings.items()}
    print("Save attribute embeddings ........")
    with open(os.path.join(arg_dict['data_dir'], arg_dict['filename'] + '_attribute_embeddings.pkl'), 'wb') as f:
        pickle.dump(attribute_embeddings, f)


    print("Generate training, validation and testing data .............")
    # prepare training, validation and testing data
    columns = [x for x in list(data.columns) if x not in meaninglessCols + ['serialized']]
    entity_ids = list(data.groupby(['entity_id']).groups.keys())
    training_entity_ids, validation_entity_ids = train_test_split(entity_ids, test_size=1.0 - arg_dict['training_ratio'], random_state=42)
    testing_entity_ids = []
    #training_entity_ids, validation_entity_ids = train_test_split(training_entity_ids, test_size=1.0 - arg_dict['training_ratio'], random_state=42)
    training_entity_dict, validation_entity_dict, testing_entity_dict = defaultdict(), defaultdict(), defaultdict()
    for eid in training_entity_ids:
        training_entity_dict[eid] = 0
    for eid in validation_entity_ids:
        validation_entity_dict[eid] = 0
    for eid in testing_entity_ids:
        testing_entity_dict[eid] = 0

    timelinessAttrs = [str(e).strip() for e in arg_dict['timelinessAttrs'].split(',')]
    training_data_np, validation_data_np, testing_data_np = [], [], []
    training_data_processed, validation_data_processed, testing_data_processed = defaultdict(list), defaultdict(list), defaultdict(list)

    # generate training, validation and testing data
    for eid, group in data.groupby(['entity_id']).groups.items():
        if eid in training_entity_dict:
            comb = itertools.combinations(list(group), 2)
            for pair in comb:
                p1, p2 = data.loc[pair[0]], data.loc[pair[1]]
                for attribute in timelinessAttrs:
                    a1, a2 = p1[['id', attribute, 'entity_id', 'timestamp']].values, p2[['id', attribute, 'entity_id', 'timestamp']].values
                    if a1[0] == a2[0]:
                        continue
                    if a1[-1] == a2[-1]:
                        continue
                    # timeliness comparison
                    if a1[-1] < a2[-1]:
                        _t_ = a1
                        a1 = a2
                        a2 = _t_
                    training_data_processed[attribute].append([a1, a2])
        elif eid in validation_entity_dict or eid in testing_entity_dict:
            for attribute in timelinessAttrs:
                ss = [data.loc[i][['id', attribute, 'entity_id', 'timestamp']].values for i in list(group)]
                ss = sorted(ss, key= lambda x : x[-1], reverse=True)
                if eid in validation_entity_dict:
                    validation_data_processed[attribute].append(ss)
                else:
                    testing_data_processed[attribute].append(ss)

    training_data_processed_, validation_data_processed_, testing_data_processed_ = [], [], []
    for k, v in training_data_processed.items():
        #v_ = [e[:-1] for e in v]
        for e in v:
            e_ = [list(z[:-2]) for z in e]
            eid = e[0][-2]
            training_data_processed_.append([k] + e_ + [eid])

    for k, v in validation_data_processed.items():
        #v_ = [e[:-1] for e in v]
        for e in v:
            e_ = [list(z[:-2]) for z in e]
            eid = e[0][-2]
            validation_data_processed_.append([k] + [e_] + [eid])

    for k, v in testing_data_processed.items():
        #v_ = [e[:-1] for e in v]
        for e in v:
            e_ = [list(z[:-2]) for z in e]
            eid = e[0][-2]
            testing_data_processed_.append([k] + [e_] + [eid])

    # sample a few temporal orders to training data
    countAllTOsValidation, countEachTOValiation, countEachValiation = 0, [], []
    for r in validation_data_processed_:
        e = r[1]
        countAllTOsValidation += (len(e)) * (len(e) - 1) // 2
        countEachTOValiation.append((len(e)) * (len(e) - 1) // 2)
        countEachValiation.append(len(e))
    more_training_num = int(arg_dict['training_ratio'] * countAllTOsValidation)
    np.random.seed(20)
    validation_sc = np.random.choice(countAllTOsValidation, more_training_num, replace=False)
    more_training_data_processed = []
    for sc in validation_sc:
        if sc < countEachTOValiation[0]:
            p_0, p_1 = int(sc / countEachValiation[0]), int(sc % countEachValiation[0])
            p0, p1 = min(p_0, p_1) + 1, max(p_0, p_1)
            more_training_data_processed.append([validation_data_processed_[0][0], validation_data_processed_[0][1][p0], validation_data_processed_[0][1][p1], validation_data_processed_[0][-1]])
            continue
        for i in range(1, len(countEachTOValiation), 1):
            if sc >= countEachTOValiation[i - 1] and sc < countEachTOValiation[i]:
                sc = sc - countEachTOValiation[i - 1]
                p_0, p_1 = int(sc / countEachValiation[i]), int(sc % countEachValiation[i])
                p0, p1 = min(p_0, p_1) + 1, max(p_0, p_1)
                more_training_data_processed.append([validation_data_processed_[i][0], validation_data_processed_[i][1][p0], validation_data_processed_[i][1][p1], validation_data_processed_[i][-1]])
                continue

    # add more training data
    training_data_processed_ += more_training_data_processed

    # generate training embeddings
    sentence_embeddings = torch.load(os.path.join(arg_dict['data_dir'], arg_dict['filename'] + '_sentence_embeddings.pt'))
    with open(os.path.join(arg_dict['data_dir'], arg_dict['filename'] + '_attribute_embeddings.pkl'), 'rb') as f:
        attribute_embeddings = pickle.load(f)
    training_embedds = []
    for index, values in enumerate(training_data_processed_):
        attribute = values[0]
        print('index : ', index)
        pos_context_index, pos_att = values[1][0], str(values[1][1])
        neg_context_index, neg_att = values[2][0], str(values[2][1])

        attribute_emb = attribute_embeddings[attribute]
        pos_context_emb = sentence_embeddings[pos_context_index]
        pos_att_emb = attribute_embeddings[pos_att]
        neg_context_emb = sentence_embeddings[neg_context_index]
        neg_att_emb = attribute_embeddings[neg_att]

        attribute_emb = torch.cat([attribute_emb, attribute_emb], 0)
        pos_instance = torch.cat([pos_context_emb, pos_att_emb], 0)
        neg_instance = torch.cat([neg_context_emb, neg_att_emb], 0)

        training_embedds.append([attribute_emb, pos_instance, neg_instance])

    # 3. save training, validation and testing data
    with open(os.path.join(arg_dict['data_dir'], 'training_processed.pkl'), 'wb') as f:
        pickle.dump(training_data_processed_, f)
    with open(os.path.join(arg_dict['data_dir'], 'validation_processed.pkl'), 'wb') as f:
        pickle.dump(validation_data_processed_, f)
    with open(os.path.join(arg_dict['data_dir'], 'testing_processed.pkl'), 'wb') as f:
        pickle.dump(testing_data_processed_, f)
    torch.save(training_embedds, os.path.join(arg_dict['data_dir'], 'training_embedded.pt'))



if __name__ == '__main__':
    main()


