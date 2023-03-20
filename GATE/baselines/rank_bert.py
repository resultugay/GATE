# %%
import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
import pickle
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score   
# from torchmetrics.functional import retrieval_reciprocal_rank, retrieval_normalized_dcg

# %%
# data prepare
def isTrainingKey(t0, t1, attr):
    return str(t0) + '---' + str(t1) + '---' + str(attr)
import os
import sys
dataset_type = sys.argv[1] #'nba' #sys.argv[1]
gpuOption = sys.argv[2] #'0' #sys.argv[2]
gamma = float(sys.argv[3])
if len(sys.argv) > 4:
    D_T = float(sys.argv[4])

model_checkpoint = 'distilbert-base-uncased'

data_file = os.path.join('../data/', dataset_type, 'data.csv')
train_file = os.path.join('../data/', dataset_type, 'training_processed.pkl')
val_file = os.path.join('../data/', dataset_type, 'validation_processed.pkl')

os.environ["CUDA_VISIBLE_DEVICES"] = gpuOption #"2"
os.environ['WANDB_DISABLED'] = 'true'

# %%
# training data: half attributes of one entity are relevant, and the remaining ones aree non-relevant
import pickle
from collections import defaultdict

with open(train_file, 'rb') as f:
    train_data = pickle.load(f)

with open(val_file, 'rb') as f:
    val_data = pickle.load(f)

np.random.seed(30)
sample_ratio = 0.5
sc = np.random.choice(len(train_data), int(sample_ratio * len(train_data)), replace=False)
print(len(sc))
train_data_ = []
for e in sc:
    ll = train_data[e]
    if ll[1][0] <= ll[2][0]:
        continue
    train_data_.append(ll)

train_data = train_data_
print(len(train_data))

# varying Gamma
sc = np.arange(len(train_data))
np.random.seed(42)
np.random.shuffle(sc)
num = int(gamma * len(sc))
sc = sc[:num]
_list_processed = []
for e in sc:
    _list_processed.append(train_data[e])
train_data = _list_processed

ifInTrainMap = defaultdict()
for record in train_data:
    attr, t0, t1 = record[0], record[1], record[2]
    key = isTrainingKey(t0, t1, attr)
    key_ = isTrainingKey(t1, t0, attr)
    ifInTrainMap[key] = 0
    ifInTrainMap[key_] = 0

print(train_data[:5])

# %%
# transform training data to the original format
train_data_dict = defaultdict(list)
for record in train_data:
    train_data_dict[(record[0], record[-1])].append(record[1])
    train_data_dict[(record[0], record[-1])].append(record[2])

#print(train_data_dict)

for key_ in train_data_dict:
    records = sorted(train_data_dict[key_], key = lambda x : x[0], reverse=1)
    records_ = [records[0]]
    for i in range(1, len(records), 1):
        if str(records_[-1]) == str(records[i]):
            continue
        records_.append(records[i])
    train_data_dict[key_] = records_


# %%


# %%
def serialize(row,cols):
    context = ''
    for c in cols:
        if c not in ['id','entity_id','row_id','timestamp']:
            context += ' [COL] ' + str(c) + ' [VAL] ' +  str(row[c])
    
    return context

data = pd.read_csv(data_file)
data['serialized'] = data.apply(serialize, axis=1,args=([data.columns]))
data_np = data['serialized'].values

# %%
train_data = []
relevant_rate = 0.2
for k, v in train_data_dict.items():
    attr, eid = k[0], k[1]
    #print(k, v)
    relevant_sc = int(len(v) * relevant_rate)
    for sc in range(relevant_sc):
        train_data.append([attr + ' [SEP] ' + data_np[v[sc][0]], 1])
    for sc in range(relevant_sc, len(v), 1):
        train_data.append([attr + ' [SEP] ' + data_np[v[sc][0]], 0])

train_data[:5]

# %%
val_data_serial = []
for record in val_data:
    record_ = []
    attr, eid = record[0], record[-1]
    for r in record[1]:
        record_.append(attr + ' [SEP] ' + data_np[r[0]])
    val_data_serial.append(record_)

# %%
from datasets import Dataset, DatasetDict

train_data_pd = pd.DataFrame(train_data, columns=['serialized', 'label'])
dataset = Dataset.from_pandas(train_data_pd)

# %%
dataset = DatasetDict({'train': dataset, 'validation': dataset})

# %%
from transformers import AutoTokenizer
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# %%
def preprocess_function(examples):
    return tokenizer(examples['serialized'], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset

# %%
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# %%
metric = 'accuracy'

args = TrainingArguments(output_dir='./rank_bert/' + dataset_type,
                        evaluation_strategy='epoch',
                        save_strategy='epoch',
                        learning_rate=2e-5,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        num_train_epochs=6,
                        weight_decay=0.01,
                        save_total_limit=2,
                        load_best_model_at_end=True,
                        metric_for_best_model='accuracy',
                        push_to_hub=False)

from sklearn import metrics
import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions)
    acc = metrics.accuracy_score(labels, predictions)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc}

trainer = Trainer(model, args, train_dataset=encoded_dataset['train'], eval_dataset=encoded_dataset['validation'], tokenizer=tokenizer, compute_metrics=compute_metrics)

import time
overall_start_time = time.time()

# %%
trainer.train()

overall_train_Time = time.time() - overall_start_time

# %%
trainer.evaluate()

# %%
# predict order
from transformers import TextClassificationPipeline

# pipe = TextClassificationPipeline(model=model.to('cpu'), tokenizer=tokenizer)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# print(val_data_serial[0])
# # define input and transfer to device
# encoding = tokenizer.encode_plus(val_data_serial[0], 
#      add_special_tokens=True, 
#      truncation=True, 
#      padding="max_length", 
#      return_attention_mask=True, 
#      return_tensors="np")

# encoding = encoding.to(device)

prediction = pipe(val_data_serial[0], return_all_scores=True)
print('PREDICTION : ', prediction)


val_data_len = []
val_data_serial_reduce = []
for vals_eid in val_data_serial:
    val_data_serial_reduce += [e[:512] for e in vals_eid]
    val_data_len.append(len(vals_eid))

predictions_all_reduce = pipe(val_data_serial_reduce, return_all_scores=True)
start = 0
predictions_all = []
for vals_eid, _num in zip(val_data_serial, val_data_len):
    predictions_all.append(predictions_all_reduce[start: start + _num])
    start += _num


# predictions_all = []
# for vals_eid in val_data_serial:
#     vals_eid_ = [[e] for e in vals_eid]
#     print('inference with ', vals_eid_)
#     predictions_all.append(pipe(torch.tensor(vals_eid_).to(device), return_all_scores=True))





overall_time = time.time() - overall_start_time

# %%
def predict_rank(predictions_json, record):
    predictions = []
    for i, ll in enumerate(predictions_json):
        predictions.append([record[i][0], ll[1]['score']])
        # if ll['label'] == 'LABEL_1':
        #     predictions.append(ll['score'])
        # else:
        #     predictions.append(1 - ll['score'])
    #predictions = [[tid, e] for tid, e in enumerate(predictions)]
    return sorted(predictions, key=lambda x: x[1], reverse=1)

# %%
predictions_order_ml = []
for predict, record in zip(predictions_all, val_data):
    predictions_order_ml.append(predict_rank(predict, record[1]))

print('predictions order ML : ', predictions_order_ml)

# %%


''' ground_truth = [3, 5, 7, 10, 2] => [1, 2, 3, 4, 5]
    prediction = [3, 5, 10, 7, 2] => [1, 2, 4, 3, 5]
'''
def metrics(prediction_order, ground_truth_order, attribute, ifInTrainMap, _seed=20):
    '''
    m = defaultdict(int)
    ground_truth = np.arange(len(ground_truth_order)) + 1
    prediction = []
    for i, tid in enumerate(ground_truth_order):
         m[tid] = ground_truth[i]
    pred = [m[tid] for tid in prediction_order]
    '''
    tidsMap = defaultdict()
    tidsMapReverse = defaultdict()
    for i, e in enumerate(ground_truth_order):
        tidsMap[e] = i
        tidsMapReverse[i] = e
    ground_truth = [0] * len(ground_truth_order)
    for i, tid in enumerate(ground_truth_order):
        ground_truth[tidsMap[tid]] = np.exp(len(ground_truth_order) - i - 1) - 1
    pred = [0] * len(ground_truth_order)
    for i, tid in enumerate(prediction_order):
        pred[tidsMap[tid]] = np.exp(len(prediction_order) - i - 1) - 1

    # NDCG
    ndcgmetric = 0 #retrieval_normalized_dcg(torch.tensor(pred, dtype=torch.float32),
                    #                      torch.tensor(ground_truth))
    # MRR
    mrr_pred = [0] * len(pred)
    #mrr_pred[np.argmin(pred)] = 1
    mrr_pred[np.argmax(pred)] = 1
    mrr_ground_truth = [True] + [False] * (len(pred) - 1)
    mrr_pred = torch.tensor(mrr_pred, dtype=torch.float32)
    mrr_ground_truth = torch.tensor(mrr_ground_truth)
    mrr = 0 #retrieval_reciprocal_rank(mrr_pred, mrr_ground_truth)

    # TP, FP, FN
    np.random.seed(_seed)
    np.random.shuffle(pred)
    np.random.seed(_seed)
    np.random.shuffle(ground_truth)
    TP, FP, FN, acc, count = 0, 0, 0, 0, 0

    ground_truth_new = [[tidsMapReverse[i], e] for i, e in enumerate(ground_truth)]
    np.random.seed(_seed)
    np.random.shuffle(ground_truth_new)

    # only compare the latest one with others
    latest_sc = np.argmax(ground_truth)
    for i in range(len(ground_truth)):
        if i == latest_sc:
            continue

        sc0, sc1 = i, latest_sc

        ttid_0, ttid_1 = ground_truth_new[sc0][0], ground_truth_new[sc1][0]
        if isTrainingKey(ttid_0, ttid_1, attribute) in ifInTrainMap:
            continue


        if i > latest_sc:
            z_ = sc0
            sc0 = sc1
            sc1 = z_
        label, prediction = ground_truth[sc0] < ground_truth[sc1], pred[sc0] < pred[sc1]
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1

    '''
    for pair in itertools.combinations(np.arange(len(pred)), 2):
        sc0, sc1 = pair[0], pair[1]
        if sc0 == sc1:
            continue
        label, prediction = ground_truth[sc0] < ground_truth[sc1], pred[sc0] < pred[sc1]
        # TP
        if label == True and prediction == True:
            TP += 1
        # FP
        if prediction == True and label == False:
            FP += 1
        # FN
        if label == True and prediction == False:
            FN += 1
        # acc
        if label == prediction:
            acc += 1
        count += 1
    '''
    return ndcgmetric, mrr, TP, FP, FN, acc, count

# %%
ndcgmetric, mrr, TP, FP, FN, acc, count = 0, 0, 0, 0, 0, 0, 0
for index, _list in enumerate(val_data):
    attr, eid = _list[0], _list[-1]

    ground_truth_order = [e[0] for e in _list[1]]
    prediction_order_ml = predictions_order_ml[index]
    prediction_order_ml = [e[0] for e in prediction_order_ml]

    print('EVAl : ', prediction_order_ml, ground_truth_order)
    ndcg_s, mrr_s, TP_s, FP_s, FN_s, acc_s, count_s = metrics(prediction_order_ml, ground_truth_order, attr, ifInTrainMap, index * 100)
    ndcgmetric += ndcg_s
    mrr += mrr_s
    TP += TP_s
    FP += FP_s
    FN += FN_s
    acc += acc_s
    count += count_s

ndcg = ndcgmetric * 1.0 / len(val_data)
mrr = mrr * 1.0 / len(val_data)
precision = TP * 1.0 / (TP + FP)
recall = TP * 1.0 / (TP + FN)
Fmeasure = 2 * precision * recall / (precision + recall)
accuracy = acc * 1.0 / count #len(validation_processed)

round = 0
print("roundGATE={} ndcg={} mrr={} precision={} recall={} Fmeasure={} accuracy={}".format(round, ndcg, mrr, precision, recall, Fmeasure, accuracy))
print('The overall time is {} and the training time is {}'.format(overall_time, overall_train_Time))

