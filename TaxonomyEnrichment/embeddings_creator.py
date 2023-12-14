from tqdm import tqdm
import os

SAVING_DIR = '/home/data/taxonomy'
os.environ["TRANSFORMERS_CACHE"] = SAVING_DIR + "hf_cache/"
os.environ["HF_HOME"] = SAVING_DIR + "hf_cache/"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import torch
import networkx as nx

device = 'cuda'

def divide_chunks(l, n): 
    
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
        
def get_model_embeddings(out, tokens):
    masked_outputs = (out['last_hidden_state'] * tokens['attention_mask'].unsqueeze(-1))
    #result = (masked_outputs.sum(dim=-2) / tokens['attention_mask'].sum(dim=1).unsqueeze(-1)).cpu().numpy()
    result = masked_outputs[:,0,:].cpu().numpy()
    return result


if __name__ == '__main__':

    # model = BertModel.from_pretrained("bert-base-uncased").to(device)
    # model.eval()
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)


    G = nx.read_edgelist('./data/MAG_CS/all.edgelist', create_using=nx.DiGraph, delimiter='\t')

    nodes = list(G.nodes)
    bs = 32

    all_embeddings = {}


    for batch in tqdm(divide_chunks(nodes, bs)):
        tokens = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            out = model(**tokens)

        embeddings = get_model_embeddings(out, tokens)

        for i, word in enumerate(batch):
            all_embeddings[word] = embeddings[i]
        
    assert len(all_embeddings.keys()) == len(G.nodes())

    with open('./data/MAG_CS/embeddings_bert.pickle', 'wb') as f:
        pickle.dump(all_embeddings, f)

    