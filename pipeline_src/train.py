import torch
from torch import nn

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from config.config import TaskConfig
import numpy as np
from trainer.train_epoch import train_epoch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from logger.logger import WanDBWriter

torch.manual_seed(57)
torch.cuda.manual_seed(57)
torch.cuda.manual_seed_all(57)
np.random.seed(57)
torch.backends.cudnn.deterministic = True


# https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer


class CustomScheduler:
    def __init__(self, model_size, optimizer, warmup, factor=2):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._step = 0

    def rate(self, step):
        return (
            1
            / self.factor
            * (
                self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5))
            )
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._step += 1
        rate = self.rate(self._step)
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()


def train(model, tknz, sampler, scheduler, criterion, logger, config):
    for epoch in range(config.n_epochs):
        print(f"Start of the epoch {epoch}")
        train_epoch(model, tknz, scheduler, sampler, criterion, logger, config, epoch)


if __name__ == "__main__":
    # create config
    config = TaskConfig()

    # model
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_checkpoint).to(
        config.device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_checkpoint,
        max_length=config.max_length,
        block_size=config.block_size,
    )

    # optmizations
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = CustomScheduler(config.emb_dim, optimizer, config.warmup)

    # wandb
    logger = WanDBWriter(config)

    # training
    if config.mode == "train":
        train(model, tokenizer, sampler, scheduler, criterion, logger, config)
    else:
        print("Unknown mode")
