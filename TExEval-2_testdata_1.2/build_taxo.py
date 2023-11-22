import yaml
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import pickle
import numpy as np
import networkx as nx

with open(r"./configs/build_taxo.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)


class TaxonomyBuilder:
    def __init__(self, root, all_verteces, max_iter=10000):
        self.root = root
        self.all_verteces = all_verteces
        self.max_iter = max_iter

    def build_taxonomy(self, strategy, **kwargs):
        self.edge_collector = getattr(self, strategy)
        self.collector_params = kwargs

        # self.pbar = tqdm(total=34000)
        self.all_edges = []
        self.i = 0
        self.build_tree(self.root, self.all_verteces)
        # self.pbar.close()

    #   return self.all_edges

    def build_tree(self, root, possible_verteces):
        top_edges_idx = self.edge_collector(
            root, possible_verteces, **self.collector_params
        )
        new_pos_verteces = np.delete(possible_verteces, top_edges_idx)
        for new_edge_idx in top_edges_idx:
            self.all_edges.append((root, possible_verteces[new_edge_idx]))
            # self.pbar.update(1)
            self.i += 1
            if self.i > self.max_iter:
                break
            self.build_tree(possible_verteces[new_edge_idx], new_pos_verteces)

    @staticmethod
    def ppl_thr_collector(root, possible_verteces, **kwargs):
        ppls = np.array(
            [kwargs["ppl_pairs"][(root, vertex)] for vertex in possible_verteces]
        )
        return np.where(np.array(ppls) < kwargs["thr"])[0]

    @staticmethod
    def ppl_top_collector(root, possible_verteces, **kwargs):
        ppls = np.array(
            [kwargs["ppl_pairs"][(root, vertex)] for vertex in possible_verteces]
        )
        return np.argsort(ppls)[: min(kwargs["top_k"], len(ppls))]


def clean_dict(pairs, use_lemma, reverse):
    new_pairs = {}
    for key, val in pairs.items():
        if use_lemma:
            term = key[0].split("(")[0].strip()
        else:
            term = key[0]
        target = key[1].split(",")[0]
        new_key = (target, term) if reverse else (term, target)
        new_pairs[new_key] = val

    return new_pairs

def delete_all_multiple_parents(G):
    for node in G.nodes():
        if G.in_degree(node) >= 5:
            edges_q = list(G.in_edges(node))
            for edge in edges_q:
                G.remove_edge(*edge)

def ppl_resolution(G, n):
    for node in G.nodes():
        if G.in_degree(node) >= n:
            edges_q = G.in_edges(node)
            ppls = {} 
            for edge in edges_q:
                weight = G[edge[0]][edge[1]]['weight']
                ppls[edge] = weight

            max_ppl_key = min(ppls, key=ppls.get) 
            edges_q = list(edges_q)
            edges_q.remove(max_ppl_key)
            G.remove_edges_from(edges_q)
    return G

def synset_resolution(G, **kwargs):
    for node in G.nodes():
        if G.in_degree(node) >= kwargs['n']:
            pairs = kwargs['helper'][node]
            for pair in pairs:
                if kwargs['ppl_compare'][pair] > kwargs['mix_thr']:
                    parents = pair[0].split('_')
                    edge1, edge2 = (parents[0], pair[1]), (parents[1], pair[1])
                    if edge2 in G.in_edges(node) and edge1 in G.in_edges(node):
                        if G[parents[0]][node]['weight'] > G[parents[1]][node]['weight']:
                            G.remove_edge(*edge1)
                        else:
                            G.remove_edge(*edge2)


def resolve_multiple_parents(G, **kwargs):
    if kwargs['enable_mixing']:
        synset_resolution(G, ppl_compare=kwargs['ppl_compare'],
                                    helper=kwargs['helper'], mix_thr=kwargs['mix_thr'], n=kwargs['n'])
    else:
        ppl_resolution(G, kwargs['n'])


def iterative_child(ppl_pairs, low, high, step, max_iter):
    thrs = np.arange(low, high, step)
    Fs = []
    for thr in tqdm(thrs):
        tb = TaxonomyBuilder(root, all_verteces, max_iter)
        tb.build_taxonomy("ppl_thr_collector", ppl_pairs=ppl_pairs, thr=thr)
        edges = tb.all_edges

        P = len(set(G.edges()) & set(edges)) / (len(set(edges)) + 1)
        R = len(set(G.edges()) & set(edges)) / len(set(G.edges()))
        F = (2 * P * R) / (P + R + 1e-15)

        #  print('precision: {} \n recall: {} \n F-score: {}'.format(P,R,F))
        Fs.append(F)

    print(max(Fs), thrs[np.argmax(Fs)])
   # plt.plot(thrs, Fs)
    return Fs

def resolve_cycle(cur_G, cycle):
    cycle_ppls = {}

    for u,v in cycle:
        val = cur_G[u][v]['weight']
        cycle_ppls[(u,v)] = val

    highest_ppl = sorted(cycle_ppls.items(), key = lambda x: x[1], reverse=True)[0][0]
    cur_G.remove_edge(*highest_ppl)

def resolve_graph_cycles(G_pred):
    while True:
        try:
            cycle = nx.find_cycle(G_pred)
            resolve_cycle(G_pred, cycle)
        except nx.NetworkXNoCycle:
            break


def brute_child(ppl_pairs, ppl_compare, helper, 
                enable_mixing, low, high, step, n):
    thrs = np.arange(low, high, step)
    Fs = []
    for thr in tqdm(thrs):
        for mix_thr in thrs:
            G_pred = get_graph(ppl_pairs, thr)

            resolve_graph_cycles(G_pred)
            resolve_multiple_parents(G_pred, enable_mixing=enable_mixing, ppl_compare=ppl_compare,
                                        helper=helper, mix_thr=mix_thr, n=n)


            P = len(set(G.edges()) & set(G_pred.edges())) / (len(set(G_pred.edges())) + 1e-15)
            R = len(set(G.edges()) & set(G_pred.edges())) / len(set(G.edges()))
            # print(len(set(edges)))
            F = (2 * P * R) / (P + R + 1e-15)

            Fs.append(F)

    print(max(Fs))#, thrs[np.argmax(Fs)])
 #   plt.plot(thrs, Fs)
    return Fs

def get_graph(ppl_pairs, thr):
    S = nx.DiGraph()
    for key, val in ppl_pairs.items():
        if val <thr:
            S.add_edge(key[0], key[1], weight=val)
    return S

def helping_dict(compare):
        """
        new view for mixes dataset {node: pair_from_ppl_compare}
        """
        helper = {}

        for i in compare:
            if i[1] not in helper:
                helper[i[1]] = [i]
            else:
                helper[i[1]].append(i)
        return helper

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

    if data == 'food':
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

    root = data
    all_verteces = list(G.nodes)
    all_verteces.remove(root)

    #  print(ppls_pairs)
    res = brute_child(ppls_pairs, ppl_compare, helper, 
                      enable_mixing=enable_mixing, low=low, 
                      high=high, step=step, n=n)
  #  res = iterative_child(ppls_pairs, low=low, high=high, step=step, max_iter=25000)
