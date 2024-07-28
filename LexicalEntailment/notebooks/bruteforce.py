import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.contrib.concurrent import process_map
import itertools


def count_diff(data, ppls_clean, low_thr=-100000, high_thr=100000):
    y_pred = []

    for child, parent, label in data:
        if not (child, parent) in ppls_clean.keys():
            y_pred.append(0)
            continue

        forward_ppl = ppls_clean[(child, parent)]
        backward_ppl = ppls_clean[(parent, child)]

        y_pred.append(np.clip(backward_ppl - forward_ppl, low_thr, high_thr))
    y_pred = preprocessing.normalize(np.array([y_pred]))[0]
    roc_auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    # print('ROC AUC score: ',)
    # print('Average precision: ',)
    return roc_auc, ap


if __name__ == "__main__":
    perplexions_path = "../data/ant_pairs_formated_def_ppls.pickle"
    test_dataset_path = "../data/ant_test.txt"

    with open(perplexions_path, "rb") as f:  # (ребенок, родитель): перплексия
        ppls = pickle.load(f)

    ppls_clean = dict()
    for item in ppls.items():
        ppls_clean[
            (item[0][0].split("(")[0].strip(", "), item[0][1].strip(", "))
        ] = item[1]

    pairs = []
    y_true = []
    non_reversed = []
    not_found = []
    data = []

    with open(test_dataset_path, "r", encoding="utf-8") as f:
        i = 0
        lines = f.readlines()
        for line in lines:
            ex1, ex2, category = line.strip("\n").split("\t")
            s11, v1, s12 = ex1.split(",")
            s21, v2, s22 = ex2.split(",")
            # if s11 == s21 and s12 == s22:
            v1 = v1.strip(" ")
            v2 = v2.strip(" ")
            if category == "directional_entailment":  # child, parent
                data.append((v1, v2, 1))

            elif category == "directional_non-entailment":  # parent, child
                data.append((v1, v2, 0))
            # else:
            #     non_reversed.append((s11, s12, v1, s21, s22, v2, category))

    y_true = [elem[2] for elem in data]

    lthr = np.arange(-1000, 0, 10)
    hthr = np.arange(0, 1000, 10)

    all_thrs_iterator = itertools.product(lthr, hthr)
    all_thrs = []
    for l, h in all_thrs_iterator:
        if l < h:
            all_thrs.append((l, h))
    print(len(all_thrs))

    out = []

    def get_metric(thrs):
        l, r = thrs
        r, a = count_diff(data, ppls_clean, low_thr=l, high_thr=h)
        return r, a

    for i in range(1, len(all_thrs) // 1000):
        cur_min = (i - 1) * 1000
        cur_max = i * 1000
        out.extend((process_map(get_metric, all_thrs[cur_min:cur_max], chunksize=1)))

    max_r = np.max([elem[0] for elem in out])
    max_p = np.max([elem[1] for elem in out])

    print(max_r, max_p)
