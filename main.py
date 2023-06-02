# %load_ext autoreload
# %autoreload 2

import os
import yaml

with open(r"params.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)


os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    map(str, params_list["CUDA_VISIBLE_DEVICES"])
)
import sys
import torch
import pandas as pd
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import wandb

sys.path.append("../NLP-DL-Project-hypo-to-hyper/pipeline_src/")


from config.config import TaskConfig
from train import CustomScheduler, train
from logger.logger import WanDBWriter
from trainer.train_epoch import train_epoch, predict
from metrics.metrics import get_all_metrics
from dataset.dataset import init_data
from logger.logger import WanDBWriter


if torch.cuda.is_available():
    device = "cuda"
    print("GPU")
else:
    device = "cpu"
    print("CPU")


SEED = params_list["SEED"][0]
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
print(torch.cuda.device_count())

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

if __name__ == "__main__":
    config = TaskConfig()

    config.n_epochs = params_list["EPOCHS"][0]
    config.batch_size = params_list["BATCH_SIZE"][0]
    config.lr = float(params_list["LR"][0])
    config.min_lr = float(params_list["MIN_LR"][0])

    config.data_path = params_list["DATA_PATH"][0]
    config.gold_path = (
        None  # "SemEval2018-Task9/training/gold/1A.english.training.gold.txt"
    )
    config.test_data_path = params_list["TEST_DATA_PATH"][0]
    config.test_gold_path = (
        None  # "SemEval2018-Task9/test/gold/1A.english.test.gold.txt"
    )

    config.device = device
    config.using_peft = params_list["USING_PEFT"][0]
    config.model_type = params_list["MODEL_TYPE"][0]  # Auto or Llama
    config.wandb_log_dir = "/raid/rabikov/wandb/"
    config.model_checkpoint = params_list["MODEL_CHECKPOINT"][0]
    config.exp_name = config.model_checkpoint.replace("/", "-")
    config.saving_path = (
        "/raid/rabikov/model_checkpoints/" + config.exp_name + "_custom"
    )

    if config.model_type == "Auto":
        model_type = AutoModelForCausalLM
        tokenizer_type = AutoTokenizer
    elif config.model_type == "Llama":
        model_type = LlamaForCausalLM
        tokenizer_type = LlamaTokenizer

    if params_list["DTYPE"][0] == "half":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = model_type.from_pretrained(
        config.model_checkpoint,
        # load_in_8bit=True,
        device_map="auto",
        torch_dtype=dtype,
    )

    tokenizer = tokenizer_type.from_pretrained(
        config.model_checkpoint,
        padding_side="left",
    )

    if config.using_peft:
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

    train_dataset, test_dataset, train_loader, val_loader = init_data(tokenizer, config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
    )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=config.lr, steps_per_epoch=len(train_loader), epochs=config.n_epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * config.n_epochs, eta_min=config.min_lr
    )

    logger = WanDBWriter(config)

    train(
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        logger,
        config,
    )
