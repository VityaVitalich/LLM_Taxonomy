import os
import yaml
import sys

with open(r"./configs/embeddings.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

SAVING_DIR = os.environ.get("SAVING_DIR")
HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["TRANSFORMERS_CACHE"] = SAVING_DIR + "hf_cache/"
os.environ["HF_HOME"] = SAVING_DIR + "hf_cache/"

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    map(str, params_list["CUDA_VISIBLE_DEVICES"])
)


import pickle
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import torch
import networkx as nx

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    LlamaTokenizer,
    LlamaForCausalLM,
    LlamaModel,
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

sys.path.append("../pipeline_src/")
from dataset.dataset import HypernymDataset, Collator
from dataset.prompt_schemas import (
    hypo_term_hyper,
    predict_child_from_2_parents,
    predict_child_from_parent,
    predict_child_with_parent_and_grandparent,
    predict_children_with_parent_and_brothers,
    predict_parent_from_child_granparent,
    predict_parent_from_child,
)

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


def get_term(s):
    term = s.split("|")[-2]
    term = term.split(":")
    return term[-1].strip()


class PplDataset(HypernymDataset):
    def __init__(
        self,
        data,
        tokenizer,
        tokenizer_encode_args={"return_tensors": "pt"},
        transforms={
            "only_child_leaf": predict_parent_from_child_granparent,
            "only_leafs_all": predict_child_from_parent,
            "only_leafs_divided": predict_children_with_parent_and_brothers,
            "leafs_and_no_leafs": predict_child_from_parent,
            "simple_triplet_grandparent": predict_parent_from_child_granparent,
            "simple_triplet_2parent": predict_child_from_2_parents,
            "predict_hypernym": predict_parent_from_child,
        },
    ):
        self.tokenizer = tokenizer
        self.tokenizer_encode_args = tokenizer_encode_args
        self.data = data
        self.case2transform = transforms


def get_model_embeddings(out, att_mask, strategy):
    ### MEAN OF TOKENS
    if strategy == "mean":
        masked_outputs = out["last_hidden_state"] * att_mask.unsqueeze(-1)
        result = (
            (masked_outputs.sum(dim=-2) / att_mask.sum(dim=1).unsqueeze(-1))
            .cpu()
            .float()
            .numpy()
        )

    ### FIRST TOKEN VECTOR
    elif strategy == "first":
        masked_outputs = (out["last_hidden_state"] * att_mask.unsqueeze(-1)).float()
        result = masked_outputs[:, 0, :].cpu().numpy()

    ### LAST TOKEN VECTOR
    elif strategy == "last":
        seq_len = att_mask.sum(dim=1)
        result = (
            out["last_hidden_state"][:, seq_len - 1, :]
            .diagonal()
            .T.float()
            .cpu()
            .numpy()
        )

    return result


if __name__ == "__main__":
    config = {}
    config["batch_size"] = params_list["BATCH_SIZE"][0]
    config["pre-train"] = params_list["USING_PRETRAIN"][0]
    config["model_checkpoint"] = params_list["MODEL_CHECKPOINT"][0]
    config["in_path"] = params_list["INPUT_PATH"][0]
    config["out_path"] = params_list["OUTPUT_PATH"][0]
    config["load_path"] = (
        SAVING_DIR + "model_checkpoints/" + params_list["LOAD_PATH"][0]
    )
    config["embedding_strategy"] = params_list["EMBEDDING_STRATEGY"][0]
    model_type = params_list["MODEL_TYPE"][0]  # Auto or Llama

    if model_type == "Auto":
        model_type = AutoModelForCausalLM
        tokenizer_type = AutoTokenizer
    elif model_type == "Llama":
        model_type = LlamaForCausalLM
        tokenizer_type = LlamaTokenizer

    extra_model_params = {}

    if config["pre-train"]:
        extra_model_params["torch_dtype"] = torch.bfloat16
        extra_model_params["load_in_4bit"] = True

    model = model_type.from_pretrained(
        config["model_checkpoint"],
        device_map="auto",
        use_auth_token=HF_TOKEN,
        **extra_model_params
    ).eval()

    tokenizer = tokenizer_type.from_pretrained(
        config["model_checkpoint"],
        padding_side="left",
        use_auth_token=HF_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if config["pre-train"]:
        LORA_R = 8
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        LORA_TARGET_MODULES = [
            "q",
            "v",
        ]

        # model = prepare_model_for_int8_training(model)
        config_lora = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            # target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config_lora)
        model.print_trainable_parameters()

        checkpoint = torch.load(config["load_path"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        torch.cuda.empty_cache()

        model = model.base_model.model.model

    with open(config["in_path"], "rb") as f:
        all_nodes = pickle.load(f)

    dataset = PplDataset(all_nodes, tokenizer)
    collator = Collator(tokenizer.eos_token_id, tokenizer.eos_token_id, -100)

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        collate_fn=collator,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=False,
    )
    all_embeddings = {}

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch
        decoded_terms = tokenizer.batch_decode(terms)
        cur_terms = list(map(get_term, decoded_terms))

        with torch.no_grad():
            out = model.forward(
                terms.to(device).long(),
                attention_mask=att_mask_terms.to(device).long(),
            )

        cur_embeddings = get_model_embeddings(
            out, att_mask_terms, config["embedding_strategy"]
        )
        for i, word in enumerate(cur_terms):
            all_embeddings[word] = cur_embeddings[i, :]

    assert len(all_embeddings.keys()) == len(all_nodes)

    with open(config["out_path"], "wb") as f:
        pickle.dump(all_embeddings, f)
