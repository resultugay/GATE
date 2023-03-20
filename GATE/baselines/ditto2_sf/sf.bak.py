from matcher import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

sys.path.insert(0, "Snippext_public")

task_config = None
model = None
testset_file = None
testset_g = None
lm = None

train_lines = []
valid_lines = []
test_lines = []

train_file = ""
valid_file = ""
test_file = ""


def eval_classifier_and_retrieve_topK(sentence_pairs, config, model, lm='distilbert', max_len=256):
    import torch.nn as nn
    import sklearn.metrics as metrics

    print("qiu: my test---")
    t1 = time.time()

    inputs = []
    for (sentA, sentB, label) in sentence_pairs:
        inputs.append(sentA + '\t' + sentB + '\t' + label)

    # dataset = DittoDataset(inputs, config['vocab'], config['name'], lm=lm, max_len=max_len)
    dataset = testset_g
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=64,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=DittoDataset.pad)

    # prediction
    Y_logits = []
    Y_hat = []
    Y = []
    Y_prob = []

    loss_list = []
    total_size = 0

    # model.eval()
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            taskname = taskname[0]
            # print("taskname:" + taskname)
            """
            logits, _, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)
            Y_logits += logits.cpu().numpy().tolist()
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            """
            logits, y1, y_hat = model(x, y, task=taskname)
            logits = logits.view(-1, logits.shape[-1])
            y1 = y1.view(-1)

            loss = nn.CrossEntropyLoss()(logits, y1)

            loss_list.append(loss.item() * y.shape[0])
            total_size += y.shape[0]

            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            Y_prob.extend(logits.softmax(dim=-1).max(dim=-1)[0].cpu().numpy().tolist())

    results = []
    for i in range(len(Y_hat)):
        pred = dataset.idx2tag[Y_hat[i]]
        results.append(pred)

    """
    qiu: 这里计算准确率
    """
    loss = sum(loss_list) / total_size

    # qiu 0811
    # print("*** qiu: print Y_prob:")
    # print(Y_prob)

    print("=============%s==================" % taskname)

    num_classes = len(set(Y))
    # Binary classification
    if num_classes <= 2:
        # qiu: Y指y_true,即输入值；Y_hat指y_pred即预测值。把Y_hat输出即可。
        # 但这里是训练的、不是测试的。
        # print("y_true:" + str(Y))
        # print("y_pred:" + str(Y_hat))
        accuracy = metrics.accuracy_score(Y, Y_hat)
        precision = metrics.precision_score(Y, Y_hat)
        recall = metrics.recall_score(Y, Y_hat)
        f1 = metrics.f1_score(Y, Y_hat)

        print("accuracy=%.3f" % accuracy)
        print("precision=%.3f" % precision)
        print("recall=%.3f" % recall)
        print("f1=%.3f" % f1)
        print("======================================")

        # return accuracy, precision, recall, f1, loss
    else:
        accuracy = metrics.accuracy_score(Y, Y_hat)
        f1 = metrics.f1_score(Y, Y_hat, average='macro')
        precision = recall = accuracy  # We might just not return anything
        print("accuracy=%.3f" % accuracy)
        print("macro_f1=%.3f" % f1)
        print("======================================")

    t2 = time.time()
    print("test time:" + str(t2 - t1))
    return results, Y_prob, accuracy


def run_train(run_id=0):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--task", type=str, default="ditto")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    # parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true", default=True)
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    # parser.add_argument("--lm", type=str, default='distilbert')
    # parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--lm", type=str, default='roberta')
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

    # config = {'name': 'Structured/Beer', 'task_type': 'classification', 'vocab': ['0', '1'],
    config = {'name': 'ditto', 'task_type': 'classification', 'vocab': ['0', '1'],
              'trainset': 'data/sics_literature_matcher_cleaned/train.txt',
              'validset': 'data/sics_literature_matcher_cleaned/val.txt',
              'testset': 'data/sics_literature_matcher_cleaned/test.txt'}

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


""" 主动学习部分 """
iteration = 1
for iter_id in range(iteration):
    run_train(iter_id)

    sentence_pairs = []
    for line in open(testset_file, 'r', encoding='utf-8'):
        sentence_pairs.append((line.replace("\n", "").split("\t")[0],
                               line.replace("\n", "").split("\t")[1], line.replace("\n", "").split("\t")[2]))

    # eval_classifier_and_retrieve_topK (sentence_pairs, task_config, model, lm, max_len=256)
    r1, r2, r3 = eval_classifier_and_retrieve_topK(sentence_pairs, task_config, model, lm, max_len=256)

    print("**** active learning iteration: " + str(iter_id) + " accuracy=" + str(r3))

    weights = {}
    for i in range(len(r2)):
        logits = r2[i]
        weights[i] = abs(logits - 0.5)

    weights = sorted(weights.items(), key=lambda asd: asd[1], reverse=True)
    new_test_lines = []
    topk = 20

    for i in range(len(weights)):
        item = weights[i]
        print(str(item[1]) + "\t" + str(r2[item[0]]) + "\t" + sentence_pairs[item[0]][0] + "\t" + sentence_pairs[item[0]][1])

