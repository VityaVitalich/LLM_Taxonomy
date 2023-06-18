import os
import yaml

with open(r"params_metrics.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

import sys
import pandas as pd
import numpy as np
import pickle


sys.path.append("pipeline_src/")

from metrics.metrics import Metric
from dataset.dataset import HypernymDataset
from transformers import AutoTokenizer

from dataset.prompt_schemas import (
    hypo_term_hyper,
    predict_child_from_2_parents,
    predict_child_from_parent,
    predict_child_with_parent_and_grandparent,
    predict_children_with_parent_and_brothers,
    predict_parent_from_child_granparent,
)

if __name__ == "__main__":
    test_path = params_list["TEST_DATA_PATH"][0]
    saving_path = "/raid/rabikov/model_outputs/" + params_list["OUTPUT_NAME"][0]

    df = pd.read_pickle(test_path)

    transforms = {
        "only_child_leaf": predict_parent_from_child_granparent,
        "only_leafs_all": predict_child_from_parent,
        "only_leafs_divided": predict_children_with_parent_and_brothers,
        "leafs_and_no_leafs": predict_child_from_parent,
        "simple_triplet_grandparent": predict_parent_from_child_granparent,
        "simple_triplet_2parent": predict_child_from_2_parents,
    }

    with open(saving_path, "rb") as fp:
        all_preds = pickle.load(fp)

    if isinstance(all_preds[0][0], list):
        flat_list = [item for sublist in all_preds for item in sublist]
        all_preds = flat_list

    all_labels = []
    all_terms = []
    cased = {}

    for i, elem in enumerate(df):
        try:
            all_preds[i]
        except IndexError:
            continue

        case = elem["case"]
        processed_term, target = transforms[case](elem)
        all_labels.append(target)
        all_terms.append(processed_term)

        if not case in cased.keys():
            cased[case] = {"pred": [], "label": [], "term": []}

        cased[case]["pred"].append(all_preds[i])
        cased[case]["label"].append(target)
        cased[case]["term"].append(processed_term)

    print("total preds:" + str(len(all_preds)))
    print("total labels:" + str(len(all_labels)))
    metric_counter = Metric(all_labels, all_preds)
    mean_cased = metric_counter.get_metrics()

    cased_metrics = {}
    for key in cased.keys():
        metric_counter = Metric(cased[key]["label"], cased[key]["pred"])
        res = metric_counter.get_metrics()
        cased_metrics[key] = res

    write_log_path = "/raid/rabikov/metrics/" + params_list["OUTPUT_NAME"][0] + ".txt"

    with open(write_log_path, "w") as f:
        df = pd.concat(
            [pd.DataFrame(cased_metrics), (pd.DataFrame(mean_cased, index=["mean"]).T)],
            axis=1,
        )
        f.write(df.to_string())
        print("written in file")
