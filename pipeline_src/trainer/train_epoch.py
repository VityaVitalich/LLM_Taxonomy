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
import itertools
from collections import Counter
import pickle


def train_epoch(
    model,
    tokenizer,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    crit,
    logger,
    config,
    epoch,
):
    # unfreeze(model)
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, batch in pbar:
        st = logger.get_step() + 1
        logger.set_step(step=st, mode="train")

        terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch

        output = model.forward(
            input_seqs.to(config.device),
            attention_mask=att_mask_input.to(config.device),
            labels=labels.to(config.device),
        )

        optimizer.zero_grad()
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.add_scalar("loss", loss.item())
        pbar.set_postfix({"Loss": loss.item()})

        if config.loss_tol != 0 and loss.item() <= config.loss_tol:
            break

    return None
    # return loss ...


@torch.no_grad()
def validate(model, val_loader, logger, config):
    model.eval()

    for batch_idx, batch in enumerate(val_loader):
        terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch

        with torch.no_grad():
            output = model.forward(
                input_seqs.to(config.device),
                attention_mask=att_mask_input.to(config.device),
                labels=labels.to(config.device),
            )
            loss = output["loss"]
            logger.add_scalar("Val_loss", loss.item())

            # del y, batch, output, loss


@torch.no_grad()
def predict(model, tokenizer, val_loader, config, epoch=""):
    model.eval()

    all_preds = []
    all_labels = []

    saving_path = config.saving_predictions_path + config.exp_name + "_" + str(epoch)

    evalbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="eval going")
    for batch_idx, batch in evalbar:
        terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch

        output_tokens = model.generate(
            terms.to(config.device),
            attention_mask=att_mask_terms.to(config.device),
            pad_token_id=tokenizer.eos_token_id,
            **config.gen_args,
        )
        pred_tokens = output_tokens[:, terms.size()[1] :]
        pred_str = tokenizer.batch_decode(pred_tokens.cpu(), skip_special_tokens=True)
        gold_str = tokenizer.batch_decode(targets, skip_special_tokens=True)

        all_preds.extend(pred_str)
        all_labels.extend(gold_str)

        if batch_idx % 10 == 0:
            with open(saving_path, "wb") as fp:
                pickle.dump(all_preds, fp)

    with open(saving_path, "wb") as fp:
        pickle.dump(all_preds, fp)
    return all_preds, all_labels


@torch.no_grad()
def predict_multiple(model, tokenizer, val_loader, config):
    """
    !!! ONLY WORKS WITH BATCH SIZE 1 !!!
    NEED TO FIX!
    """
    model.eval()

    all_preds = []
    all_labels = []

    evalbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="eval going")
    for batch_idx, batch in evalbar:
        terms, att_mask_terms, targets, input_seqs, att_mask_input, labels = batch

        output_tokens = model.generate(
            terms.to(config.device),
            attention_mask=att_mask_terms.to(config.device),
            pad_token_id=tokenizer.eos_token_id,
            **config.gen_args,
        )
        pred_tokens = output_tokens[:, terms.size()[1] :]
        pred_str = tokenizer.batch_decode(pred_tokens.cpu(), skip_special_tokens=True)
        gold_str = tokenizer.batch_decode(targets, skip_special_tokens=True)

        del output_tokens
        torch.cuda.empty_cache()

        merged_iter = itertools.chain.from_iterable(
            list(map(lambda x: x.split(","), pred_str))
        )
        sorted_predicted_answer = [
            i[0].strip().replace("\n", "") for i in Counter(merged_iter).most_common()
        ]

        all_preds.append(sorted_predicted_answer)
        all_labels.extend(gold_str)

        if batch_idx % 5 == 0:
            with open("/raid/rabikov/model_outputs/predictions_dolly_2", "wb") as fp:
                pickle.dump(all_preds, fp)

    return all_preds, all_labels