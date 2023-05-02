from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
import gc
from utils.plotters import visualize_predictions
import wandb
from utils.tools import freeze, unfreeze
import json
from metrics.metrics import get_all_metrics


def train_epoch(
    model,
    tokenizer,
    scheduler,
    train_loader,
    val_loader,
    crit,
    logger,
    config,
    epoch,
):
    unfreeze(model)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, batch in pbar:
        st = logger.get_step() + 1
        logger.set_step(step=st, mode="train")

        terms, targets, input_seqs, labels = batch

        output = model.forward(
            input_seqs.to(config.device), labels=labels.to(config.device)
        )

        scheduler.zero_grad()
        loss = output["loss"]
        loss.backward()
        scheduler.step()

        logger.add_scalar("loss", loss.item())
        pbar.set_postfix({"Loss": loss.item()})

        if config.loss_tol != 0 and loss.item() <= config.loss_tol:
            break

        # кажется пересылки и удаления очень очень едят время
        # del y, batch, output, loss
        # gc.collect()
        # torch.cuda.empty_cache()

        if (batch_idx + 1) % config.validation == 0:
            validate(model, val_loader, logger, config)

        if (batch_idx + 1) % config.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    # 'opt': optimizer.state_dict(),
                    "sch": scheduler.state_dict(),
                },
                f"best_model_{st}_{epoch}.pth",
            )

        if (batch_idx + 1) % config.show_every == 0:
            # show some examples TODO
            # visualize_predictions()
            # и это тоже пихнуть в валидацию
            pass

        if (batch_idx + 1) % config.compute_metrics_every == 0:
            # кмк это можно пихнуть в валидацию чтобы снизить
            # вычисления и метрики смотреть
            all_preds, all_labels = predict(model, val_loader, tokenizer, config)
            metrics = get_all_metrics(all_labels, all_preds)

            for key in metrics:
                logger.add_scalar(key, metrics[key])

    return None
    # return loss ...


@torch.no_grad()
def validate(model, val_loader, logger, config):
    freeze(model)

    for batch_idx, batch in enumerate(val_loader):
        terms, targets, input_seqs, labels = batch

        with torch.no_grad():
            output = model.forward(
                input_seqs.to(config.device), labels=labels.to(config.device)
            )
            loss = output["loss"]
            logger.add_scalar("Val_loss", loss.item())

            # del y, batch, output, loss


@torch.no_grad()
def predict(model, tokenizer, val_loader, config):
    freeze(model)

    all_preds = []
    all_labels = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for batch_idx, batch in pbar:
        terms, targets, input_seqs, labels = batch

        output_tokens = model.generate(
            terms.to(config.device),
            pad_token_id=tokenizer.eos_token_id,
            **config.gen_args,
        )
        pred_tokens = output_tokens[:, terms.size()[1] :]
        pred_str = tokenizer.batch_decode(pred_tokens.cpu(), skip_special_tokens=True)
        gold_str = tokenizer.batch_decode(targets, skip_special_tokens=True)

        all_preds.extend(pred_str)
        all_labels.extend(gold_str)

    return all_preds, all_labels
