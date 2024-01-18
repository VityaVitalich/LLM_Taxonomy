import yaml
import os

with open(r"./configs/definitions.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

import pickle 
import networkx as nx
import wikipedia
import nltk
import wikipediaapi
from nltk.tokenize import sent_tokenize, word_tokenize
from Levenshtein import distance

from tqdm import tqdm


def get_right_names(node):
    try:
        return wikipedia.search(node, results=0, suggestion=False)[0]
    except:
        return ''

def get_summary(name, wiki_wiki):
    page_py = wiki_wiki.page(name)
    print
    return ' '.join(sent_tokenize(page_py.summary)[:3])

if __name__ == '__main__':
    
    wiki_wiki = wikipediaapi.Wikipedia('TaxRes (esneminova@gmail.com)', 'en')
    config = {}
    config["input_path"] = params_list["INPUT_PATH"][0]
    config["output_path"] = params_list["OUTPUT_PATH"][0]
    config["intermediate_path"] = params_list["INTERMEDIATE_PATH"][0]

    # getting appropriate names
    if os.path.exists(config["intermediate_path"]):
        names = pickle.load(open(config["intermediate_path"], 'rb'))
    else:
        G = nx.read_edgelist(config["input_path"], create_using=nx.DiGraph, delimiter="\t")
        nodes = list(G.nodes)
        print(nodes)

        names = {}
        for i in tqdm(nodes):
            print(i)
            names[i] = get_right_names(i)
        pickle.dump(names, open(config["intermediate_path"], 'wb'))

    # getting definitions
    new_dict = {}
    for i in tqdm(names):
        distance_len = distance(i, names[i].lower())
        if distance_len < 3 and names[i] != '':
            new_dict[i] = get_summary(names[i], wiki_wiki)

        elif distance_len > 10 and names[i] != '':
            new_dict[i] = get_summary(names[i], wiki_wiki)

        else:
            new_dict[i] = ''

    pickle.dump(new_dict, open(config['output_path'], 'wb'))