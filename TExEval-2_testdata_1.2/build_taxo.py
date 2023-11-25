import yaml
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import pickle
import numpy as np
import networkx as nx
from build_utils import clean_dict, brute_child, iterative_child
from multiparent_refinment import helping_dict
from cycle_refinment import clean_triplets
from conflict_refinment import clean_triplets_conflicts

with open(r"./configs/build_taxo.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)


if __name__ == "__main__":
    data = params_list["DATA"][0]
    in_name = params_list["IN_NAME"][0]
    compare_name = params_list["COMPARE_NAME"][0]
    enable_mixing = params_list["ENABLE_MIXING"][0]
    lemma = params_list["LEMMA"][0]
    reverse = params_list["REVERSE"][0]
    n = params_list["N_PARENTS"][0]
    low = params_list["LOW"][0]
    high = params_list["HIGH"][0]
    step = params_list["STEP"][0]
    insertions_path = params_list["INSERTIONS_PATH"][0]
    resolve_conflicts = params_list["RESOLVE_CONFLICTS"][0]
    hypo_path = params_list["HYPO_PATH"][0]
    use_insertion = params_list["USE_INSERTION_CONFLICT"][0]

    if data == "food":
        path = "gs_taxo/EN/" + str(data) + "_wordnet_en.taxo"
    else:
        path = "gs_taxo/EN/" + str(data) + "_eurovoc_en.taxo"
    G = nx.DiGraph()

    with open(path, "r") as f:
        for line in f:
            idx, hypo, hyper = line.split("\t")
            hyper = hyper.replace("\n", "")
            G.add_node(hypo)
            G.add_node(hyper)
            G.add_edge(hyper, hypo)

    with open(in_name, "rb") as f:
        ppls = pickle.load(f)

    ppls_pairs = clean_dict(ppls, use_lemma=lemma, reverse=reverse)

    with open(compare_name, "rb") as f1:
        ppls_c = pickle.load(f1)

    ppl_compare = clean_dict(ppls_c, use_lemma=lemma, reverse=False)
    helper = helping_dict(ppl_compare)

    with open(insertions_path, "rb") as f:
        insertions_raw = pickle.load(f)
    insertions = clean_triplets(insertions_raw)
    insertions_conflict = clean_triplets_conflicts(insertions_raw)

    with open(hypo_path, "rb") as f:
        ppls_hypo = pickle.load(f)
    ppls_hypo = clean_dict(ppls, use_lemma=lemma, reverse=False)

    root = data
    all_verteces = list(G.nodes)
    all_verteces.remove(root)

    #  print(ppls_pairs)
    build_args = {
        "G": G,
        "ppl_pairs": ppls_pairs,
        "low": low,
        "high": high,
        "step": step,
    }

    mix_parents_args = {
        "ppl_compare": ppl_compare,
        "helper": helper,
        "enable_mixing": enable_mixing,
        "n": n,
    }

    cycles_args = {
        "insertions": insertions,
    }

    conflict_args = {
        "resolve_conflicts": resolve_conflicts,
        "insertions_conflict": insertions_conflict,
        "ppls_hypo": ppls_hypo,
        "use_insertion": use_insertion,
    }

    build_args.update(mix_parents_args)
    build_args.update(cycles_args)
    build_args.update(conflict_args)

    res = brute_child(**build_args)
#  res = iterative_child(ppls_pairs, low=low, high=high, step=step, max_iter=25000)
