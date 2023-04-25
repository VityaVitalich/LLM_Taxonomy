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
from metrics.metrics import mean_average_precision


def train_iter(model, tknz, sched, sampler, crit, logger, config, epoch):
    tokenizer = tknz
    criterion = crit
    # optimizer = optim
    scheduler = sched
    
    total_steps = config.steps
    
    for i in tqdm(range(total_steps)):
        
        unfreeze(model)
        
        st = logger.get_step()+1
        logger.set_step(step=st, mode="train")

        batch, y = sampler.sample(config.batch_size)

        output = model(batch)
        
        scheduler.zero_grad()
        loss = criterion(batch, y)
        loss.backward()
        scheduler.step()

        logger.add_scalar('loss', loss.item())

        if config.loss_tol != 0 and loss.item() <= config.loss_tol:
            break

        del y, batch, output, loss
        gc.collect()
        torch.cuda.empty_cache()
        
        if (i + 1) % config.validation == 0:
            freeze(model)
            
            batch, y = sampler.sample(config.batch_size)
            
            with torch.no_grad():
                output = model(batch)
                loss = criterion(output, y)
                logger.add_scalar('Val_loss', loss.item())
                del y, batch, output, loss

        if (i + 1) % config.save_every == 0:
            torch.save({
                    'model': model.state_dict(),
                    # 'opt': optimizer.state_dict(),
                    'sch': scheduler.state_dict()
            }, f"best_model_{st}.pth")

            
        if (i + 1) % config.show_every == 0:

            # show some examples TODO
            # visualize_predictions()
            pass
            
            
        if (i + 1) % config.compute_metrics_every == 0:
            # TODO
            batch, y = sampler.sample(config.batch_size)
            
            with torch.no_grad():
                output = model(batch)
                map_ = mean_average_precision(output, y)
                logger.add_scalar('MAP', map_)
                del y, batch, output, map_
    
    return None
    # return loss ...