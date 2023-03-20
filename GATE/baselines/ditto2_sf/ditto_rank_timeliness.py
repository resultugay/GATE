import os
import sys
from collections import defaultdict
import time
import argparse


def isTrainingKey(t0, t1, attr):
    return str(t0) + '---' + str(t1) + '---' + str(attr)

parser = argparse.ArgumentParser()
# parser.add_argument("--task", type=str, default="Structured/Beer")
parser.add_argument("--task", type=str, default="ditto")
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--lr", type=float, default=3e-5)
# parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--n_epochs", type=int, default=3)
parser.add_argument("--finetuning", dest="finetuning", action="store_true", default=True)
parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)
parser.add_argument("--logdir", type=str, default="checkpoints/")
parser.add_argument("--lm", type=str, default='distilbert')
# parser.add_argument("--lm", type=str, default='roberta')
# parser.add_argument("--lm", type=str, default='bert-base-uncased')
parser.add_argument("--bert_path", type=str, default=None)
parser.add_argument("--fp16", dest="fp16", action="store_true")
parser.add_argument("--da", type=str, default='swap')
parser.add_argument("--alpha_aug", type=float, default=0.8)
parser.add_argument("--dk", type=str, default=None)
parser.add_argument("--summarize", dest="summarize", action="store_true")
parser.add_argument("--balance", dest="balance", action="store_true")
parser.add_argument("--size", type=int, default=None)
parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", default=True)
parser.add_argument('--dataset_type', type=str, default='nba')
parser.add_argument('--gpuOption', type=str, default='')
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--D_T', type=float, default=1.0)

hp = parser.parse_args()


def varyDT(data_processed, dt):
    sc = np.arange(len(data_processed))
    np.random.seed(42)
    np.random.shuffle(sc)
    num = int(dt * len(sc))
    sc = sc[:num]
    _list_processed = []
    for e in sc:
        _list_processed.append(data_processed[e])
    return _list_processed


meaninglessCols = ['id', 'entity_id', 'row_id', 'timestamp']

dataset_type = hp.dataset_type #'career' #sys.argv[1]
gpuOption = hp.gpuOption #'0'

data_file = os.path.join('../../../data/', dataset_type, 'data.csv')
train_file = os.path.join('../../../data/', dataset_type, 'training_processed.pkl')
val_file = os.path.join('../../../data/', dataset_type, 'validation_processed.pkl')

os.environ["CUDA_VISIBLE_DEVICES"] = gpuOption #"2"
# data_file = "career/data.csv"
# train_file = "career/training_processed.pkl"
# val_file = "career/validation_processed.pkl"
num_epochs = 1 # 5

import random

run_id = random.randint(0, 10000)
out_file = "out_" + str(run_id) + ".txt"
out_pkl = "out_" + str(run_id) + ".pkl"

prediction_file = out_pkl

# 第一步，读表文件，截取其中文本

texts = []

schemas = []
for line in open(data_file, 'r', encoding='utf-8'):
    schemas = line
    break
schemas = [e.strip() for e in schemas.split(',')]
print(schemas)
meaninglessCols_sc = [i for i in range(len(schemas)) if schemas[i] in meaninglessCols]
print(meaninglessCols_sc)

for line in open(data_file, 'r', encoding='utf-8'):
    line = line.replace("\n", "")
    strs = line.split(",")

    text = ""
    for i, s in enumerate(strs[:len(schemas)]):
        #if not s.isdigit():
        if i not in meaninglessCols_sc:
            # text = text + s + " "
            text = text + ' [COL] ' + schemas[i] + ' [VAL] ' + s + " "

    texts.append(text)

# ddata = pd.read_csv(data_file)
# dcolumns = list(ddata.columns)
# meaninglessCols_sc = [i for i in range(len(dcolumns)) if dcolumns[i] in meaninglessCols]

# texts = []
# for line in ddata.values:
#     text = ''
#     for i, s in enumerate(line):
#         if i not in meaninglessCols_sc:
#             # text = text + s + ' '
#             text = text + ' [COL] ' + dcolumns[i] + ' [VAL] ' + s + " "
#     texts.append(text)

# 删掉表头
del (texts[0])

# 第二步，读标注训练数据

import pickle
from sentence_transformers import InputExample

# train_obj为list，每个元素的格式为：
# <class 'list'>: ['LN', [100, 'luca_changed'], [99, 'luca'], 13]

# 读取训练数据
with open(train_file, 'rb') as f:
    train_list = pickle.load(f)

import numpy as np

np.random.seed(30)
sample_ratio = 0.5
sc = np.random.choice(len(train_list), int(sample_ratio * len(train_list)), replace=False)
_list_processed = []
for e in sc:
    ll = train_list[e]
    if ll[1][0] <= ll[2][0]:
        continue
    _list_processed.append(train_list[e])
train_list = _list_processed

# varying Gamma
sc = np.arange(len(train_list))
np.random.seed(42)
np.random.shuffle(sc)
num = int(hp.gamma * len(sc))
sc = sc[:num]
_list_processed = []
for e in sc:
    _list_processed.append(train_list[e])
train_list = _list_processed


ifInTrainMap = defaultdict()
for record in train_list:
    attr, t0, t1 = record[0], record[1][0], record[2][0]
    key = isTrainingKey(t0, t1, attr)
    key_ = isTrainingKey(t1, t0, attr)
    ifInTrainMap[key] = 0
    ifInTrainMap[key_] = 0


train_nli = []
for train_data in train_list:
    line1 = train_data[1][0]
    line2 = train_data[2][0]
    train_nli.append(InputExample(texts=[texts[line1], texts[line2]], label=1))
    train_nli.append(InputExample(texts=[texts[line2], texts[line1]], label=0))

dev_nli = train_nli

out1 = open(os.path.join('./data', dataset_type, 'train.txt'), 'w', encoding='utf-8')
out2 = open(os.path.join('./data', dataset_type, 'val.txt'), 'w', encoding='utf-8')
out3 = open(os.path.join('./data', dataset_type, 'test.txt'), 'w', encoding='utf-8')

# out1 = open('data/person/train.txt', "w", encoding='utf-8')
# out2 = open('data/person/val.txt', "w", encoding='utf-8')
# out3 = open('data/person/test.txt', "w", encoding='utf-8')


def serializeT(attr, vals):
    # return attr + ' [ATTR] ' + vals
    return attr + ' ' + vals

test_number = 0
for train_data in train_list:
    line1 = serializeT(train_data[0], texts[train_data[1][0]]) #str(train_data[0]) + ' [SEP] ' + texts[train_data[1][0]]
    line2 = serializeT(train_data[0], texts[train_data[2][0]]) #str(train_data[0]) + ' [SEP] ' + texts[train_data[2][0]]
    line1 = line1.replace("\t", " ").replace("\n", " ").replace("_", " ")
    line2 = line2.replace("\t", " ").replace("\n", " ").replace("_", " ")
    out1.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
    out1.write(line2 + "\t" + line1 + "\t" + "0" + "\n")

    out2.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
    out2.write(line2 + "\t" + line1 + "\t" + "0" + "\n")
    out3.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
    out3.write(line2 + "\t" + line1 + "\t" + "0" + "\n")

    # if test_number < 100:
    #     out2.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
    #     out2.write(line2 + "\t" + line1 + "\t" + "0" + "\n")
    #     out3.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
    #     out3.write(line2 + "\t" + line1 + "\t" + "0" + "\n")
    # test_number = test_number + 1

###########################

from matcher import *
from logger import Logger
from Snippext_public.snippext.model import MultiTaskNet

log = Logger('log/out_information.txt', level='debug')

model = None

# config = {'name': 'Structured/Beer', 'task_type': 'classification', 'vocab': ['0', '1'],
'''
config = {'name': 'ditto', 'task_type': 'classification', 'vocab': ['0', '1'],
          'trainset': 'data/person/train.txt',
          'validset': 'data/person/val.txt',
          'testset': 'data/person/test.txt'}
'''
config = {'name': 'ditto', 'task_type': 'classification', 'vocab': ['0', '1'],
          'trainset': os.path.join('./data', dataset_type, 'train.txt'),
          'validset': os.path.join('./data', dataset_type, 'val.txt'),
          'testset': os.path.join('./data', dataset_type, 'test.txt')}


max_len = 256

import time
overall_start_time = time.time()

def run_train(run_id=0):
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--task", type=str, default="Structured/Beer")
    # parser.add_argument("--task", type=str, default="ditto")
    # parser.add_argument("--run_id", type=int, default=0)
    # parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--max_len", type=int, default=256)
    # parser.add_argument("--lr", type=float, default=3e-5)
    # # parser.add_argument("--lr", type=float, default=0.01)
    # parser.add_argument("--n_epochs", type=int, default=3)
    # parser.add_argument("--finetuning", dest="finetuning", action="store_true", default=True)
    # parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)
    # parser.add_argument("--logdir", type=str, default="checkpoints/")
    # # parser.add_argument("--lm", type=str, default='distilbert')
    # parser.add_argument("--lm", type=str, default='roberta')
    # # parser.add_argument("--lm", type=str, default='bert-base-uncased')
    # parser.add_argument("--bert_path", type=str, default=None)
    # parser.add_argument("--fp16", dest="fp16", action="store_true")
    # parser.add_argument("--da", type=str, default='swap')
    # parser.add_argument("--alpha_aug", type=float, default=0.8)
    # parser.add_argument("--dk", type=str, default=None)
    # parser.add_argument("--summarize", dest="summarize", action="store_true")
    # parser.add_argument("--balance", dest="balance", action="store_true")
    # parser.add_argument("--size", type=int, default=None)
    # parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")

    # hp = parser.parse_args()

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
                                                            hp.dk, hp.summarize, str(hp.size), run_id)
    run_tag = run_tag.replace('/', '_')

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    task_type = config['task_type']
    vocab = config['vocab']
    tasknames = [task]

    global train_file
    global valid_file
    global test_file

    global train_lines
    global valid_lines
    global test_lines

    train_lines = []
    valid_lines = []
    test_lines = []

    train_file = trainset
    valid_file = validset
    test_file = testset

    for line in open(train_file, 'r', encoding='utf-8'):
        train_lines.append(line)
    for line in open(valid_file, 'r', encoding='utf-8'):
        valid_lines.append(line)
    for line in open(test_file, 'r', encoding='utf-8'):
        test_lines.append(line)

    # load train/dev/test sets
    train_dataset = DittoDataset(trainset, vocab, task,
                                 lm=hp.lm,
                                 max_len=hp.max_len,
                                 size=hp.size,
                                 balance=hp.balance)
    valid_dataset = DittoDataset(validset, vocab, task, lm=hp.lm)
    test_dataset = DittoDataset(testset, vocab, task, lm=hp.lm)

    print(hp)
    log.logger.info(r'hp:{}'.format(hp))

    from snippext.mixda import initialize_and_train
    augment_dataset = DittoDataset(trainset, vocab, task,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   augment_op=hp.da,
                                   size=hp.size,
                                   balance=hp.balance)
    initialize_and_train(config,
                         train_dataset,
                         augment_dataset,
                         valid_dataset,
                         test_dataset,
                         hp,
                         run_tag)

    global task_config
    global model
    global testset_file
    global testset_g
    global lm

    lm = hp.lm
    testset_g = test_dataset
    if hp.use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    print('DEVICE : ', device)
    model = MultiTaskNet([config], device, True, lm=lm)

    saved_state = torch.load(run_tag + '_dev.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state, strict=False)

    model = model.to(device)

    task_config = config
    task_config['finetuning'] = False

    # task_config, model = load_model(hp.task, run_tag + '_dev.pt',
    #                            hp.lm, hp.use_gpu, hp.fp16)
    testset_file = testset

import random
run_id = random.randint(0,100000)
run_train(run_id)

overall_train_time = time.time() - overall_start_time

###########################

# 排序预测

model.eval()

result_list = []
out = open(out_file, "w", encoding='utf-8')
with open(val_file, 'rb') as f:
    val_list = pickle.load(f)

if hp.D_T != 1.0:
    val_list = varyDT(val_list, hp.D_T)

# transformer validation data
val_list_with_sc = []
for index, record in enumerate(val_list):
    vlist = [[i] + e for i, e in enumerate(record[1])]
    val_list_with_sc.append([record[0], vlist, record[-1]])

# val_list_with_sc = val_list_with_sc[:100]

import itertools
def sortStandard(all_number):
    return list(itertools.combinations(np.arange(all_number), 2))

# predict all pairs (This is the fastest way so far !!!)
ditto_pairs_to_cmp, ditto_tid_pairs = [], []
for index, record in enumerate(val_list_with_sc):
    attr, eid = record[0], record[-1]
    ditto_cal = []
    _num = len(record[1])
    pair_combs = sortStandard(_num)
    pairs_num = len(pair_combs)
    for sc_pair in pair_combs:
        sc_1, sc_2 = sc_pair[0], sc_pair[1]
        tid0, tid1 = record[1][sc_1][0], record[1][sc_2][0]
        input_1, input_2 = serializeT(attr, texts[tid0]), serializeT(attr, texts[tid1])
        ditto_pairs_to_cmp.append([input_1, input_2])
        ditto_tid_pairs.append([tid0, tid1])

model = model.to('cuda')
predictions_all, _ = classify(ditto_pairs_to_cmp, config, model, lm=lm, max_len=max_len)
print('!!!!!! PREDICTIOn : ')
print(predictions_all[:5])
print()
print(predictions_all[1])
print('predictions number : ', len(predictions_all), len(ditto_pairs_to_cmp))

val_list_all_predictions = []
start = 0
for index, record in enumerate(val_list_with_sc):
    attr, eid = record[0], record[-1]
    _num = len(record[1])
    pair_combs = sortStandard(_num)
    # print('PAIr : ', pair_combs, _num)
    pairs_num = len(pair_combs)
    predictions_ditto = predictions_all[start: start + pairs_num]
    tid_pairs_ditto = ditto_tid_pairs[start: start + pairs_num]
    tuple_pairs_predictions = defaultdict()
    for i, sc_pair in enumerate(pair_combs):
        # print(sc_pair)
        # k = str([sc_pair[0], sc_pair[1]])
        tid0, tid1 = tid_pairs_ditto[i]
        k_1 = str([tid0, tid1])
        tuple_pairs_predictions[k_1] = predictions_ditto[i]
        k_2 = str([tid1, tid0])
        if predictions_ditto[i] == '0':
            tuple_pairs_predictions[k_2] = '1'
        else:
            tuple_pairs_predictions[k_2] = '0'
    val_list_all_predictions.append([attr, record[1], eid, tuple_pairs_predictions])
    start += pairs_num

true_num = 0
false_num = 0
for val_data in val_list_all_predictions:
    items = val_data[1]
    np.random.seed(20)
    np.random.shuffle(items)
    print('Before : ', items)
    for i in range(1, len(items)):

        key = items[i]

        j = i - 1
        if_continue = True
        while j >= 0 and if_continue:

            #predictions, logits = classify([[texts[key[0]].replace("_", " "), texts[items[j][0]].replace("_", " ")]], config, model, lm=lm, max_len=max_len)
            # if key[0] < items[j][0]:
                # predictions = val_data[-1][str([key[0], items[j][0]])]
            # else:
                # predictions = val_data[-1][str([items[j][0], key[0]])]
            predictions = val_data[-1][str([key[0], items[j][0]])]
            print(predictions)

            # print(pred_labels)
            if predictions[0] == "0" or predictions[0] == 0:
                if_continue = False
                false_num = false_num + 1
            else:
                items[j + 1] = items[j]
                j -= 1
                true_num = true_num + 1
        items[j + 1] = key
        # items.reverse()
    print('After : ', items)
    out.write("[" + str(val_data[0]) + "," + str(items) + "," + str(val_data[2]) + "]\n")

    result = []
    result.append(val_data[0])
    # result.append(items)
    result.append([e[1:] for e in items])
    result.append(val_data[2])
    result_list.append(result)

# true_num = 0
# false_num = 0
# for val_data in val_list:
#     items = val_data[1]
#     np.random.seed(20)
#     np.random.shuffle(items)
#     for i in range(1, len(items)):

#         key = items[i]

#         j = i - 1
#         if_continue = True
#         while j >= 0 and if_continue:

#             predictions, logits = classify([[texts[key[0]].replace("_", " "), texts[items[j][0]].replace("_", " ")]], config, model, lm=lm, max_len=max_len)
#             print(predictions)

#             # print(pred_labels)
#             if predictions[0] == "0" or predictions[0] == 0:
#                 if_continue = False
#                 false_num = false_num + 1
#             else:
#                 items[j + 1] = items[j]
#                 j -= 1
#                 true_num = true_num + 1
#         items[j + 1] = key
#         # items.reverse()
#     print(items)
#     out.write("[" + str(val_data[0]) + "," + str(items) + "," + str(val_data[2]) + "]\n")

#     result = []
#     result.append(val_data[0])
#     result.append(items)
#     result.append(val_data[2])
#     result_list.append(result)

print("\n\n\n\n\n\n\n\n\n\n*****************\n")
print("true_num: " + str(true_num))
print("false_num: " + str(false_num))

print(len(val_list))
print(len(result_list))
with open(out_pkl, 'wb') as fid:
    pickle.dump(result_list, fid)

#

import itertools
from collections import defaultdict

import numpy as np
import torch
from torchmetrics.functional import retrieval_reciprocal_rank, retrieval_normalized_dcg


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
    ndcgmetric = retrieval_normalized_dcg(torch.tensor(pred, dtype=torch.float32),
                                          torch.tensor(ground_truth))
    # MRR
    mrr_pred = [0] * len(pred)
    #mrr_pred[np.argmin(pred)] = 1
    mrr_pred[np.argmax(pred)] = 1
    mrr_ground_truth = [True] + [False] * (len(pred) - 1)
    mrr_pred = torch.tensor(mrr_pred, dtype=torch.float32)
    mrr_ground_truth = torch.tensor(mrr_ground_truth)
    mrr = retrieval_reciprocal_rank(mrr_pred, mrr_ground_truth)

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


overall_time = time.time() - overall_start_time

import pickle

ground_truth_list = []
prediction_list = []

with open(val_file, 'rb') as f:
    val_list = pickle.load(f)

if hp.D_T != 1.0:
    val_list = varyDT(val_list, hp.D_T)


val_list_ = []
for record in val_list:
    if len(record[1]) <= 1:
        continue
    val_list_.append(record)
val_list = val_list_

# with open(prediction_file, 'rb') as f:
#     prediction_list_tmp = pickle.load(f)
predictions_list_tmp = [] #result_list

for record in result_list:
    if len(record[1]) <= 1:
        continue
    predictions_list_tmp.append(record)

ndcgmetric, mrr, TP, FP, FN, acc, count = 0, 0, 0, 0, 0, 0, 0
for index, _list in enumerate(val_list):
    attr, eid = _list[0], _list[-1]

    ground_truth_order = [e[0] for e in _list[1]]
    prediction_order_ml = [e[0] for e in predictions_list_tmp[index][1]]

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
print('The overall time is {} and the training time is {}'.format(overall_time, overall_train_time))







# attrs_val = []
# for i in range(len(val_list)):
#     attrs_val.append(val_list[i][0])
#     val_data = val_list[i][1]
#     prediction_data = prediction_list_tmp[i][1]

#     ground_truth = []
#     for i in range(len(val_data)):
#         key = val_data[i]
#         ground_truth.append(key[0])
#     ground_truth_list.append(ground_truth)

#     prediction = []
#     for i in range(len(prediction_data)):
#         key = prediction_data[i]
#         prediction.append(key[0])
#     prediction_list.append(prediction)

# ndcgmetric, mrr, TP, FP, FN, acc, count = 0, 0, 0, 0, 0, 0, 0

# for i in range(len(ground_truth_list)):
#     attr = attrs_val[i]
#     ground_truth = ground_truth_list[i]
#     pred = prediction_list[i]

#     order_dict = dict()
#     order = 1
#     ground_truth_order = []
#     for item in ground_truth:
#         order_dict[item] = order
#         ground_truth_order.append(order)
#         order = order + 1

#     prediction_order = []
#     for item in pred:
#         order = order_dict[item]
#         prediction_order.append(order)

#     print('Order !!! : ', prediction_order, ground_truth_order)
#     ndcg_s, mrr_s, TP_s, FP_s, FN_s, acc_s, count_s = metrics(prediction_order, ground_truth_order, attr, ifInTrainMap, i * 100)
#     ndcgmetric += ndcg_s
#     mrr += mrr_s
#     TP += TP_s
#     FP += FP_s
#     FN += FN_s
#     acc += acc_s
#     count += count_s

# ndcg = ndcgmetric * 1.0 / len(ground_truth_list)
# mrr = mrr * 1.0 / len(ground_truth_list)
# precision = TP * 1.0 / (TP + FP)
# recall = TP * 1.0 / (TP + FN)
# Fmeasure = 2 * precision * recall / (precision + recall)
# accuracy = acc * 1.0 / count

# round=0
# result_str = "round={} ndcg={} mrr={} precision={} recall={} Fmeasure={} accuracy={}".format(round, ndcg, mrr, precision,
#                                                                                  recall, Fmeasure, accuracy)
# print(result_str)
# print('The overall time is {} and the training time is {}'.format(overall_time, overall_train_time))
