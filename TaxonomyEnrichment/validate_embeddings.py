import os
import yaml
import sys

with open(r"./configs/embeddings_validate.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

from tqdm import tqdm
import pickle
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


if __name__ == "__main__":
    config = {}
    config["embedding_path"] = params_list["EMBEDDING_PATH"][0]
    config["test_path"] = params_list["TEST_PATH"][0]
    config["all_path"] = params_list["INPUT_PATH"][0]
    config["number_neighbors"] = params_list["NUM_NEIGHBORS"][0]

    with open(config["embedding_path"], "rb") as f:
        embeddings = pickle.load(f)

    with open(config["test_path"], "rb") as f:
        test_nodes = pickle.load(f)

    G_gold = nx.read_edgelist(
        config["all_path"], create_using=nx.DiGraph, delimiter="\t"
    )

    word_names = list(embeddings.keys())
    values = np.vstack(list(embeddings.values()))

    total_found = 0
    mrr = 0
    mr = 0
    nn = NearestNeighbors(
        n_neighbors=config["number_neighbors"], metric="euclidean", algorithm="brute"
    ).fit(values)
    for i in tqdm(range(1000)):
        child, parents = test_nodes[i]

        child_embedding = embeddings[child]
        dists, idxs = nn.kneighbors(child_embedding[np.newaxis, :])

        predicted_names = []
        for idx in idxs[0]:
            predicted_names.append(word_names[idx])

        found = 0
        for gt in parents:
            found += gt in predicted_names

        if found == 0:
            # print(child, ': not found')
            mrr += 0

        else:
            best_pos = 1000
            for gt in parents:
                if gt in predicted_names:
                    pos = predicted_names.index(gt)
                    if pos < best_pos:
                        best_pos = pos
            #   print(child, ': found', '; best pos = ', best_pos)

            mrr += 1 / (best_pos + 1)
            mr += best_pos + 1
            total_found += 1

    print(total_found, mrr, mrr / 1000, mr / 1000)
