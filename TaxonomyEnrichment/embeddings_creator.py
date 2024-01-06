import os
import yaml
import sys

with open(r"./configs/embeddings_bert.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

SAVING_DIR = params_list["SAVING_DIR"][0]
os.environ["TRANSFORMERS_CACHE"] = SAVING_DIR + "hf_cache/"
os.environ["HF_HOME"] = SAVING_DIR + "hf_cache/"

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    map(str, params_list["CUDA_VISIBLE_DEVICES"])
)


from tqdm import tqdm
import pickle
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import torch
import fasttext
import numpy as np
import networkx as nx

if torch.cuda.is_available():
    device = "cuda"
    print("GPU")
else:
    device = "cpu"
    print("CPU")


SEED = params_list["SEED"][0]
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
print(torch.cuda.device_count())


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def get_model_embeddings(model, batch, model_type, tokenizer=None):
    if model_type == "BERT":
        tokens = tokenizer.batch_encode_plus(
            batch, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            out = model(**tokens)
        masked_outputs = out["last_hidden_state"] * tokens["attention_mask"].unsqueeze(
            -1
        )
        # result = (masked_outputs.sum(dim=-2) / tokens['attention_mask'].sum(dim=1).unsqueeze(-1)).cpu().numpy()
        result = masked_outputs[:, 0, :].cpu().numpy()
    elif model_type == "FastText":
        result = np.vstack([model.get_sentence_vector(word) for word in batch])
    elif model_type == "SentenceBert":
        result = model.encode(batch)

    return result


if __name__ == "__main__":
    config = {}
    config["batch_size"] = params_list["BATCH_SIZE"][0]
    config["model_checkpoint"] = params_list["MODEL_CHECKPOINT"][0]
    config["in_path"] = params_list["INPUT_PATH"][0]
    config["out_path"] = params_list["OUTPUT_PATH"][0]
    model_type = params_list["MODEL_TYPE"][0]  # bert or SentenceBert of Fasttext

    if model_type == "BERT":
        model = BertModel.from_pretrained(config["model_checkpoint"]).to(device)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(config["model_checkpoint"])
    elif model_type == "SentenceBert":
        model = SentenceTransformer(config["model_checkpoint"])
        tokenizer = None
    elif model_type == "FastText":
        model = fasttext.load_model(config["model_checkpoint"])
        tokenizer = None

    G = nx.read_edgelist(config["in_path"], create_using=nx.DiGraph, delimiter="\t")

    nodes = list(G.nodes)
    bs = config["batch_size"]

    all_embeddings = {}

    for batch in tqdm(divide_chunks(nodes, bs)):
        embeddings = get_model_embeddings(model, batch, model_type, tokenizer)

        for i, word in enumerate(batch):
            all_embeddings[word] = embeddings[i]

    assert len(all_embeddings.keys()) == len(G.nodes())

    with open(config["out_path"], "wb") as f:
        pickle.dump(all_embeddings, f)
