import itertools
from collections import defaultdict

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from torchmetrics.functional import retrieval_reciprocal_rank, retrieval_normalized_dcg

''' ground_truth = [3, 5, 7, 10, 2] => [1, 2, 3, 4, 5]
    prediction = [3, 5, 10, 7, 2] => [1, 2, 4, 3, 5]
'''



def metrics(prediction_order, ground_truth_order):
    import torch
    import numpy as np

    print(prediction_order)
    print(ground_truth_order)
    print("\n")
    '''
    m = defaultdict(int)
    ground_truth = np.arange(len(ground_truth_order)) + 1
    prediction = []
    for i, tid in enumerate(ground_truth_order):
         m[tid] = ground_truth[i]
    pred = [m[tid] for tid in prediction_order]
    '''
    tidsMap = defaultdict()
    for i, e in enumerate(ground_truth_order):
        tidsMap[e] = i
    ground_truth = [0] * len(ground_truth_order)
    for i, tid in enumerate(ground_truth_order):
        ground_truth[tidsMap[tid]] = len(ground_truth_order) - i
    pred = [0] * len(ground_truth_order)
    for i, tid in enumerate(prediction_order):
        pred[tidsMap[tid]] = len(prediction_order) - i

    # NDCG
    ndcgmetric = retrieval_normalized_dcg(torch.tensor(pred, dtype=torch.float32),
                                          torch.tensor(ground_truth))
    # MRR
    mrr_pred = [0] * len(pred)
    # mrr_pred[np.argmin(pred)] = 1
    mrr_pred[np.argmax(pred)] = 1
    mrr_ground_truth = [True] + [False] * (len(pred) - 1)
    mrr_pred = torch.tensor(mrr_pred, dtype=torch.float32)
    mrr_ground_truth = torch.tensor(mrr_ground_truth)
    mrr = retrieval_reciprocal_rank(mrr_pred, mrr_ground_truth)

    # TP, FP, FN
    np.random.seed(20)
    np.random.shuffle(pred)
    np.random.seed(20)
    np.random.shuffle(ground_truth)
    TP, FP, FN = 0, 0, 0
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

    return ndcgmetric, mrr, TP, FP, FN


model = None
# config = {'name': 'Structured/Beer', 'task_type': 'classification', 'vocab': ['0', '1'],
config = {'name': 'ditto', 'task_type': 'classification', 'vocab': ['0', '1'],
          'trainset': 'data/person/train.txt',
          'validset': 'data/person/val.txt',
          'testset': 'data/person/test.txt'}
max_len = 256

from matcher import *
from logger import Logger
from Snippext_public.snippext.model import MultiTaskNet

def run_train(run_id=0, num_epochs=3):

    log = Logger('log/out_information.txt', level='debug')

    parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--task", type=str, default="ditto")
    parser.add_argument("--run_id", type=int, default=run_id)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    # parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_epochs", type=int, default=num_epochs)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true", default=True)
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    # parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--lm", type=str, default='roberta')
    # parser.add_argument("--lm", type=str, default='bert-base-uncased')
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default='swap')
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--balance", dest="balance", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")

    hp = parser.parse_args()

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
    model = MultiTaskNet([config], device, True, lm=lm)

    saved_state = torch.load(run_tag + '_dev.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state, strict=False)

    model = model.to(device)

    task_config = config
    task_config['finetuning'] = False

    # task_config, model = load_model(hp.task, run_tag + '_dev.pt',
    #                            hp.lm, hp.use_gpu, hp.fp16)
    testset_file = testset



from sentence_transformers import InputExample
from matcher import *
import pickle as pk
import random as rd

def run_score(data_file="person/data.csv", train_file="person/training_processed.pkl",
              val_file="person/validation_processed.pkl", num_epochs=3
              ):
    import numpy as np

    # data_file = "career/data.csv"
    # train_file = "career/training_processed.pkl"
    # val_file = "career/validation_processed.pkl"
    # num_epochs = 3

    run_id = rd.randint(0, 10000)
    out_file = "out_" + str(run_id) + ".txt"
    out_pkl = "out_" + str(run_id) + ".pkl"

    prediction_file = out_pkl

    # 第一步，读表文件，截取其中文本

    texts = []

    for line in open(data_file, 'r', encoding='utf-8'):
        line = line.replace("\n", "")
        strs = line.split(",")

        text = ""
        for s in strs:
            if not s.isdigit():
                text = text + s + " "

        texts.append(text)

    # 删掉表头
    del (texts[0])

    # 第二步，读标注训练数据

    # train_obj为list，每个元素的格式为：
    # <class 'list'>: ['LN', [100, 'luca_changed'], [99, 'luca'], 13]

    # 读取训练数据
    with open(train_file, 'rb') as f:
        train_list = pk.load(f)

    np.random.seed(42)
    sample_ratio = 0.1
    sc = np.random.choice(len(train_list), int(sample_ratio * len(train_list)), replace=False)
    _list_processed = []
    for e in sc:
        _list_processed.append(train_list[e])
    train_list = _list_processed

    rd.shuffle(train_list)

    train_nli = []
    for train_data in train_list:
        line1 = train_data[1][0]
        line2 = train_data[2][0]
        train_nli.append(InputExample(texts=[texts[line1], texts[line2]], label=1))
        train_nli.append(InputExample(texts=[texts[line2], texts[line1]], label=0))

    # dev_nli = train_nli
    # with open(val_file, 'rb') as f:
    #     dev_list = pk.load(f)
    #
    # np.random.seed(42)
    # sample_ratio = 0.1
    # sc = np.random.choice(len(train_list), int(sample_ratio * len(train_list)), replace=False)
    # _list_processed = []
    # for e in sc:
    #     _list_processed.append(train_list[e])
    # dev_list = _list_processed
    #
    # dev_nli = []
    # for dev_data in dev_list:
    #     line1 = dev_data[1][0]
    #     line2 = dev_data[2][0]
    #     dev_nli.append(InputExample(texts=[texts[line1], texts[line2]], label=1))
    #     dev_nli.append(InputExample(texts=[texts[line2], texts[line1]], label=0))

    out1 = open('data/person/train.txt', "w", encoding='utf-8')
    out2 = open('data/person/val.txt', "w", encoding='utf-8')
    out3 = open('data/person/test.txt', "w", encoding='utf-8')

    test_number = 0
    for train_data in train_list:
        line1 = texts[train_data[1][0]]
        line2 = texts[train_data[2][0]]
        line1 = line1.replace("\t", " ").replace("\n", " ").replace("_", " ")
        line2 = line2.replace("\t", " ").replace("\n", " ").replace("_", " ")
        out1.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
        out1.write(line2 + "\t" + line1 + "\t" + "0" + "\n")
        out2.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
        out2.write(line2 + "\t" + line1 + "\t" + "0" + "\n")
        out3.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
        out3.write(line2 + "\t" + line1 + "\t" + "0" + "\n")

    # for dev_data in dev_list:
    #     line1 = texts[dev_data[1][0]]
    #     line2 = texts[dev_data[2][0]]
    #     out2.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
    #     out2.write(line2 + "\t" + line1 + "\t" + "0" + "\n")
    #     out3.write(line1 + "\t" + line2 + "\t" + "1" + "\n")
    #     out3.write(line2 + "\t" + line1 + "\t" + "0" + "\n")


    ###########################
    run_train(run_id=run_id, num_epochs=3)

    ###########################

    # 排序预测

    model.eval()

    result_list = []
    out = open(out_file, "w", encoding='utf-8')
    with open(val_file, 'rb') as f:
        val_list = pk.load(f)

    true_num = 0
    false_num = 0

    for val_data in val_list:
        items = val_data[1]
        for i in range(1, len(items)):

            key = items[i]

            j = i - 1
            if_continue = True
            while j >= 0 and if_continue:

                predictions, logits = classify([[texts[key[0]].replace("_", " "), texts[items[j][0]].replace("_", " ")]], config, model, lm=lm, max_len=max_len)

                # print(pred_labels)
                if predictions[0] == "0":
                    if_continue = False
                    false_num = false_num + 1
                else:
                    items[j + 1] = items[j]
                    j -= 1
                    true_num = true_num + 1
            items[j + 1] = key
            # items.reverse()
        print(items)
        out.write("[" + str(val_data[0]) + "," + str(items) + "," + str(val_data[2]) + "]\n")

        result = []
        result.append(val_data[0])
        result.append(items)
        result.append(val_data[2])
        result_list.append(result)

    print("\n\n\n\n\n\n\n\n\n\n*****************\n")
    print("true_num: " + str(true_num))
    print("false_num: " + str(false_num))

    print(len(val_list))
    print(len(result_list))
    with open(out_pkl, 'wb') as fid:
        pk.dump(result_list, fid)

    ground_truth_list = []
    prediction_list = []

    with open(val_file, 'rb') as f:
        val_list = pk.load(f)

    # with open(prediction_file, 'rb') as f:
    #     prediction_list_tmp = pickle.load(f)
    prediction_list_tmp = result_list

    for i in range(len(val_list)):
        val_data = val_list[i][1]
        prediction_data = prediction_list_tmp[i][1]

        ground_truth = []
        for i in range(len(val_data)):
            key = val_data[i]
            ground_truth.append(key[0])
        ground_truth_list.append(ground_truth)

        prediction = []
        for i in range(len(prediction_data)):
            key = prediction_data[i]
            prediction.append(key[0])
        prediction_list.append(prediction)

    ndcgmetric, mrr, TP, FP, FN = 0, 0, 0, 0, 0

    for i in range(len(ground_truth_list)):
        ground_truth = ground_truth_list[i]
        pred = prediction_list[i]

        order_dict = dict()
        order = 1
        ground_truth_order = []
        for item in ground_truth:
            order_dict[item] = order
            ground_truth_order.append(order)
            order = order + 1

        prediction_order = []
        for item in pred:
            order = order_dict[item]
            prediction_order.append(order)

        ndcg_s, mrr_s, TP_s, FP_s, FN_s = metrics(prediction_order, ground_truth_order)
        ndcgmetric += ndcg_s
        mrr += mrr_s
        TP += TP_s
        FP += FP_s
        FN += FN_s

    ndcg = ndcgmetric * 1.0 / len(ground_truth_list)
    mrr = mrr * 1.0 / len(ground_truth_list)
    precision = TP * 1.0 / (TP + FP)
    recall = TP * 1.0 / (TP + FN)
    Fmeasure = 2 * precision * recall / (precision + recall)

    result_str = "round={} ndcg={} mrr={} precision={} recall={} Fmeasure={}".format(round, ndcg, mrr, precision,
                                                                                     recall, Fmeasure)
    print(result_str)
    return result_str

if __name__ == '__main__':
    s = run_score(data_file="person/data.csv", train_file="person/training_processed.pkl",
                  val_file="person/validation_processed.pkl", num_epochs=3
                  )