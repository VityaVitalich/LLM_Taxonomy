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
from trainer.train_epoch import train_epoch, predict
from metrics.metrics import get_all_metrics
from torch.utils.data import DataLoader
from dataset.dataset import HypernymDataset, Collator
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


def train(
    model, tokenizer, train_loader, val_loader, scheduler, criterion, logger, config
):
    for epoch in range(config.n_epochs):
        print(f"Start of the epoch {epoch}")
        train_epoch(
            model,
            tokenizer,
            scheduler,
            train_loader,
            val_loader,
            criterion,
            logger,
            config,
            epoch,
        )
        if config.using_peft:
             all_preds, all_labels = predict(model.model, tokenizer, val_loader, config)
        else:
            all_preds, all_labels = predict(model, tokenizer, val_loader, config)
        metrics = get_all_metrics(all_labels, all_preds)
        for key in metrics:
            logger.add_scalar(key, float(metrics[key]))

        if (epoch + 1) % config.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    # 'opt': optimizer.state_dict(),
                    # "sch": scheduler.state_dict(),
                },
                f"{config.saving_path}_epoch={epoch}_MAP={metrics['MAP']}.pth",
            )


if __name__ == "__main__":
    # create config
    config = TaskConfig()

    # model
    model = AutoModelForCausalLM.from_pretrained(config.model_checkpoint).to(
        config.device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_checkpoint,
        padding_side="left",
    )

    # data
    train_dataset = HypernymDataset(
        data_path=config.data_path,
        tokenizer=tokenizer,
        gold_path=config.gold_path,
        semeval_format=True,
    )
    test_dataset = HypernymDataset(
        data_path=config.test_data_path,
        tokenizer=tokenizer,
        gold_path=config.test_gold_path,
        semeval_format=True,
    )

    collator = Collator(tokenizer.eos_token_id, tokenizer.eos_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=8,
        drop_last=True,
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
        train(
            model,
            tokenizer,
            train_loader,
            val_loader,
            scheduler,
            criterion,
            logger,
            config,
        )
    else:
        print("Unknown mode")
