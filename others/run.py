def run_score(data_file="person/data.csv", train_file="person/training_processed.pkl",
                               val_file="person/validation_processed.pkl", num_epochs=3
                               ):
    import random
    run_id = random.randint(0,10000)
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

    import pickle
    from torch.utils.data import DataLoader
    from sentence_transformers import losses
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

    # train_obj为list，每个元素的格式为：
    # <class 'list'>: ['LN', [100, 'luca_changed'], [99, 'luca'], 13]

    # 读取训练数据
    with open(train_file, 'rb') as f:
        train_list = pickle.load(f)

    import numpy as np
    np.random.seed(42)
    sample_ratio = 0.1
    sc = np.random.choice(len(train_list), int(sample_ratio * len(train_list)), replace=False)
    _list_processed = []
    for e in sc:
        _list_processed.append(train_list[e])
    train_list = _list_processed

    train_nli = []
    for train_data in train_list:
        line1 = train_data[1][0]
        line2 = train_data[2][0]
        train_nli.append(InputExample(texts=[texts[line1], texts[line2]], label=1))
        train_nli.append(InputExample(texts=[texts[line2], texts[line1]], label=0))

    dev_nli = train_nli

    model = CrossEncoder('./bert-base-uncased')

    # qiu: test
    # num_epochs = 1
    # num_epochs = 10
    train_batch_size = 64
    # distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

    train_dataloader_nli = DataLoader(train_nli, shuffle=True, batch_size=train_batch_size)
    train_loss_nli = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=768,
                                        num_labels=2)

    dev_evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_nli, name='sts-dev')

    # Train the model
    output_nli = "output_nli"
    model.fit(
        # train_objectives=[(train_dataloader_nli, train_loss_nli)],
        train_dataloader=train_dataloader_nli,
        evaluator=dev_evaluator,
        epochs=num_epochs,
        evaluation_steps=2,
        output_path=output_nli
    )
    # 排序预测

    model.model.eval()

    result_list = []
    out = open(out_file, "w", encoding='utf-8')
    with open(val_file, 'rb') as f:
        val_list = pickle.load(f)
    for val_data in val_list:
        items = val_data[1]
        for i in range(1, len(items)):

            key = items[i]

            j = i - 1
            if_continue = True
            while j >= 0 and if_continue:
                pred_scores_1 = model.predict(
                    [[texts[key[0]], texts[items[j][0]]], [texts[items[j][0]], texts[key[0]]]], convert_to_numpy=True,
                    show_progress_bar=False)
                # pred_scores_2 = model.predict([texts[items[j][0]], texts[key[0]]], convert_to_numpy=True, show_progress_bar=False)
                # pred_labels = pred_scores_1[0] > pred_scores_1[1]
                pred_labels = pred_scores_1[0] < pred_scores_1[1]
                # print(pred_labels)
                if pred_labels:
                    if_continue = False
                else:
                    items[j + 1] = items[j]
                    j -= 1
            items[j + 1] = key
        print(items)
        out.write("[" + str(val_data[0]) + "," + str(items) + "," + str(val_data[2]) + "]\n")

        result = []
        result.append(val_data[0])
        result.append(items)
        result.append(val_data[2])
        result_list.append(result)

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

    def metrics(prediction_order, ground_truth_order):
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

    import pickle

    ground_truth_list = []
    prediction_list = []

    with open(val_file, 'rb') as f:
        val_list = pickle.load(f)

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
    res = []
    for i in range(10):
        # s = run_score(data_file="person/data.csv", train_file="person/training_processed.pkl",
        #           val_file="person/validation_processed.pkl", num_epochs=i
        #           )
        s = run_score(data_file="nba/data.csv", train_file="nba/training_processed.pkl",
                      val_file="nba/validation_processed.pkl", num_epochs=i
                      )
        res.append(s)

    print(res)
    # run_score(data_file="person/data.csv", train_file="person/training_processed.pkl",
    #           val_file="person/validation_processed.pkl", num_epochs=1
    #           )
    # run_score(data_file="nba/data.csv", train_file="nba/training_processed.pkl",
    #           val_file="nba/validation_processed.pkl", num_epochs=1
    #           )
    # run_score(data_file="career/data.csv", train_file="career/training_processed.pkl",
    #           val_file="career/validation_processed.pkl", num_epochs=1
    #           )