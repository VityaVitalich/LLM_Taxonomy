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


def train_iter_LM(
    model, tokenizer, train_loader, val_loader, scheduler, logger, config, epoch
):
    unfreeze(model)

    for batch_idx, batch in tqdm(enumerate(train_loader)):
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


def validate(model, val_loader, logger, config):
    freeze(model)

    for batch_idx, batch in tqdm(enumerate(val_loader)):
        terms, targets, input_seqs, labels = batch

        with torch.no_grad():
            output = model.foward(
                input_seqs.to(config.device), labels=labels.to(config.device)
            )
            loss = output["loss"]
            logger.add_scalar("Val_loss", loss.item())

            # del y, batch, output, loss


def predict(model, val_loader, tokenizer, config):
    freeze(model)

    all_preds = []
    all_labels = []

    for batch_idx, batch in tqdm(enumerate(val_loader)):
        terms, targets, input_seqs, labels = batch

        with torch.no_grad():
            output_tokens = model.generate(terms.to(config.device), **config.gen_args)
            pred_tokens = output_tokens[:, terms.size()[1] :]
            pred_str = tokenizer.batch_decode(
                pred_tokens.cpu(), skip_special_token=True
            )
            gold_str = tokenizer.batch_decode(labels, skip_special_token=True)

            all_preds.extend(pred_str)
            all_labels.extend(gold_str)

    return all_preds, all_labels
