import os
import csv
import time
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from model import DDG_RDE_Network, Codebook
from trainer import CrossValidation, recursive_to
from dataloader import SkempiDatasetManager
from utils import set_seed, check_dir, eval_skempi_three_modes, save_code, load_config
from torch.nn import DataParallel
from ipdb import set_trace
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


save_dir = '/home/hongtan/pretrain_single/weight'
log_file = open(os.path.join(save_dir, 'training_log2.txt'), 'a')
param = json.loads(open("/home/hongtan/pretrain_single/config/param_configs.json", 'r').read())
args = argparse.Namespace(**param)


def pretrain(epoch, model, optimizer):
    model.train()
    
    batch = recursive_to(next(iter(dataloader_manager.pretrain_loader)), device)

    _, e_q_loss, s_recon_loss, h_recon_loss, x_recon_loss = model(batch)
    loss_vae = e_q_loss * args.loss_weight + s_recon_loss + h_recon_loss + x_recon_loss

    optimizer.zero_grad()
    loss_vae.backward()
    optimizer.step()

    if epoch % 10 == 1:
        print("\033[0;30;43m{} | [pretrain] Epoch {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f} {:.8f}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, loss_vae.item(), e_q_loss.item(), s_recon_loss.item(), h_recon_loss.item(), x_recon_loss.item()))
        log_file.write("{} | [pretrain] Epoch {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f} {:.8f}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, loss_vae.item(), e_q_loss.item(), s_recon_loss.item(), h_recon_loss.item(), x_recon_loss.item()))
        log_file.flush()


vae_model = Codebook(args).to(device)
vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

dataloader_manager = SkempiDatasetManager(batch_size=32, num_workers = 4)

for epoch in range(1, args.pre_epoch+1):
    pretrain(epoch, vae_model, vae_optimizer)
    torch.cuda.empty_cache() 

for p in vae_model.parameters():
    p.requires_grad_(False)

torch.save(vae_model.state_dict(), os.path.join(save_dir, 'checkpoint', f'vae_s9683.ckpt'))
