import numpy as np
import os 
import pickle as pkl
from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pandas as pd
import math
import wandb

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def score_calculate(pred, true):
    sum = 0
    for i in range(len(pred)):
        if pred[i] >= true[i]:
            sum += math.exp((pred[i]-true[i])/10) - 1
        else:
            sum += math.exp((true[i]-pred[i])/13) - 1
    return sum 


def adjust_learning_rate(optimizer1, optimizer2, epoch, lr):
    warmup_epoch = 30
    if epoch <= warmup_epoch:
        lr = lr * (epoch+5) / (warmup_epoch+5)
    if epoch >= warmup_epoch:
        lr = lr * ((warmup_epoch+5) / (epoch+5)) ** 3
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_decay(optimizer1, optimizer2, epoch, lr):
    lr, decay_rate, decay_step = lr, 0.1, [15,25,35]
    for i in decay_step:
        if epoch >= i:
            lr *= decay_rate
        else:
            break
        
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr 
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr 
    return lr

def torch_seed(seed_num = 2):
    np.random.seed(seed_num)
    torch.manual_seed(seed_num )
    torch.cuda.manual_seed(seed_num )
    torch.backends.cudnn.benchmark=True
    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mae(predictions, targets):
    return np.abs(predictions - targets).mean()

def wandb_record(rmse, mae, score, discri_score, Context_FID):
    metrics = {
        "eva_rmse": np.mean(np.array(rmse)),
        "eva_mae": np.mean(np.array(mae)),
        "eva_score": np.mean(np.array(score)),
        "eva_discri_score": np.mean(np.array(discri_score)),
        "eva_Context_FID": np.mean(np.array(Context_FID)),
        "eva_rmse_std": np.std(np.array(rmse)),
        "eva_mae_std": np.std(np.array(mae)),
        "eva_score_std": np.std(np.array(score)),
        "eva_discri_score_std": np.std(np.array(discri_score)),
        "eva_Context_FID_std": np.std(np.array(Context_FID))
    }

    for key, value in metrics.items():
        wandb.run.summary[key] = value    
