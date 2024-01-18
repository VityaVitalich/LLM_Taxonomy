import yaml
import os

with open(r"./configs/definitions.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

import pickle 
import networkx as nx
import wikipedia
import nltk
import string
import wikipediaapi
from nltk.tokenize import sent_tokenize, word_tokenize
from Levenshtein import distance

from tqdm import tqdm


def get_right_names(node):
    try:
        most_relevant = wikipedia.search(node, results=5, suggestion=False)
        dists = {}
        for j in most_relevant:
            dists[j] = distance(node, j.lower())
        return min(dists, key=dists.get)
    except:
        return ''

def get_summary(name, wiki_wiki):
    page_py = wiki_wiki.page(name)
    print
    return ' '.join(sent_tokenize(page_py.summary)[:3])

if __name__ == '__main__':
    
    wiki_wiki = wikipediaapi.Wikipedia('TaxRes (esneminova@gmail.com)', 'en', timeout=200.0)
    config = {}
    config["input_path"] = params_list["INPUT_PATH"][0]
    config["output_path"] = params_list["OUTPUT_PATH"][0]
    config["intermediate_path"] = params_list["INTERMEDIATE_PATH"][0]

    # getting appropriate names
    if os.path.exists(config["intermediate_path"]):
        print('Loading precomputed names...')
        names = pickle.load(open(config["intermediate_path"], 'rb'))
        print('Done!')

    else:
        G = nx.read_edgelist(config["input_path"], create_using=nx.DiGraph, delimiter="\t")
        nodes = list(G.nodes)
        # print(nodes)

        names = {}
        for i in tqdm(nodes):
            # print(i)
            names[i] = get_right_names(i)
            
        pickle.dump(names, open(config["intermediate_path"], 'wb'))

    # getting definitions
    new_dict = {}
    counter = 0
    for i in tqdm(names):
        counter += 1
        # distance_len = distance(i, names[i].lower())
        if names[i] != '':
            to_source = names[i].translate(str.maketrans(string.punctuation + '–', ' '*len(string.punctuation + '–'))).lower()
            if counter % 1000 == 0:
                print(i, to_source)
            if to_source == i:
                try:
                    new_dict[i] = get_summary(names[i], wiki_wiki)
                except:
                    new_dict[i] = 'catched en error'
            else:
                new_dict[i] = ''
        else:
            new_dict[i] = ''

        # if distance_len < 3 
        # and names[i] != '':
        #     # print(i, names[i])
        #     new_dict[i] = get_summary(names[i], wiki_wiki)
        #     # print(new_dict[i])
        # elif distance_len > 10 and names[i] != '':
        #     new_dict[i] = get_summary(names[i], wiki_wiki)
        # else:
        #     new_dict[i] = ''

    pickle.dump(new_dict, open(config['output_path'], 'wb'))