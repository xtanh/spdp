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
from torch.utils.data import DataLoader

from model import DDG_RDE_Network, Codebook
from trainer import CrossValidation, recursive_to
from utils import set_seed, check_dir, eval_skempi_three_modes, save_code, load_config
from dataset import SkempiDataset
from torch.utils.data import random_split
from ipdb import set_trace

import math
from torch.utils.data._utils.collate import default_collate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_nb': -1, 
    'chain_id': ' ', 
}

class PaddingCollate(object):
    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):

        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        max_length = math.ceil(max_length / 8) * 8
        keys = self._get_common_keys(data_list)


        data_list_padded = []
        for data in data_list:
            data_padded = {k: self._pad_last(v, max_length, value=self._get_pad_value(k)) for k, v in data.items() if k in keys}
            data_list_padded.append(data_padded)

        batch = default_collate(data_list_padded)
        return batch



import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm.auto import tqdm


# RMSE: Root Mean Squared Error
def compute_rmse(true_vals, pred_vals):
    mse = mean_squared_error(true_vals, pred_vals)
    return np.sqrt(mse)

# PCC: Pearson correlation coefficient
def compute_pcc(true_vals, pred_vals):
    return pearsonr(true_vals, pred_vals)[0]

# R²: Coefficient of determination (R-squared)
def compute_r2(true_vals, pred_vals):
    return r2_score(true_vals, pred_vals)


def train_model(model, train_loader, val_loader, optimizer, epochs=50, device='cuda:0'):
    model = model.to(device)
    best_val_loss = float('inf')
    best_model_path = '/home/hongtan/pretrain_single/weight/best_model.pth'
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training loop with progress bar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', dynamic_ncols=True) as pbar:
            for batch in train_loader:
                batch = recursive_to(batch, device)
                optimizer.zero_grad()

                loss, _ = model(batch,vae_model)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'Train Loss': loss.item()})
                pbar.update(1)

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}')
        
        # Validation loop
        val_rmse, val_pcc, val_r2, avg_val_loss = validate_model(model, val_loader, device)
        print(f'Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val PCC: {val_pcc:.4f}, Val R²: {val_r2:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Val Loss: {avg_val_loss:.4f}")

def validate_model(model, val_loader, device):
    model.eval()
    
    all_ddg_true = []
    all_ddg_pred = []
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', dynamic_ncols=True):
            batch = recursive_to(batch, device)
            
            # Forward pass
            loss, output_dict = model(batch,vae_model)
            val_loss += loss.item()

            # Collect true and predicted ddG values
            ddg_true = output_dict['ddG_true'].cpu().numpy()
            ddg_pred = output_dict['ddG_pred'].cpu().numpy()
            
            all_ddg_true.append(ddg_true)
            all_ddg_pred.append(ddg_pred)

    # Flatten lists of ddG values
    all_ddg_true = np.concatenate(all_ddg_true)
    all_ddg_pred = np.concatenate(all_ddg_pred)
    avg_val_loss = val_loss / len(val_loader)
    
    # Compute RMSE, PCC, and R²
    rmse = compute_rmse(all_ddg_true, all_ddg_pred)
    pcc = compute_pcc(all_ddg_true, all_ddg_pred)
    r2 = compute_r2(all_ddg_true, all_ddg_pred)
    
    return rmse, pcc, r2, avg_val_loss

def test_model(model, test_loader, device='cuda:0'):
    model.eval()
    all_ddg_true = []
    all_ddg_pred = []
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing', dynamic_ncols=True):
            batch = recursive_to(batch, device)
            
            # Forward pass
            loss, output_dict = model(batch,vae_model)
            test_loss += loss.item()

            # Collect true and predicted ddG values
            ddg_true = output_dict['ddG_true'].cpu().numpy()
            ddg_pred = output_dict['ddG_pred'].cpu().numpy()
            
            all_ddg_true.append(ddg_true)
            all_ddg_pred.append(ddg_pred)

    # Flatten lists of ddG values
    all_ddg_true = np.concatenate(all_ddg_true)
    all_ddg_pred = np.concatenate(all_ddg_pred)
    avg_test_loss = test_loss / len(test_loader)

    # Compute RMSE, PCC, and R²
    rmse = compute_rmse(all_ddg_true, all_ddg_pred)
    pcc = compute_pcc(all_ddg_true, all_ddg_pred)
    r2 = compute_r2(all_ddg_true, all_ddg_pred)
    
    print(f'Test Loss: {avg_test_loss:.4f}, Test RMSE: {rmse:.4f}, Test PCC: {pcc:.4f}, Test R²: {r2:.4f}')
    return rmse, pcc, r2, avg_test_loss

param = json.loads(open("/home/hongtan/pretrain_single/config/param_configs.json", 'r').read())
args = argparse.Namespace(**param)


#dataset:
dataset = SkempiDataset(
            csv_path="/home/hongtan/pretrain_single/data/output_file.csv",
            pdb_dir="/home/hongtan/pretrain_single/data/pdb_s2815",
            cache_dir="/home/hongtan/pretrain_single/data/cache",
            patch_size=128
        )
s276_dataset = SkempiDataset(
            csv_path="/home/hongtan/pretrain_single/data/dataset/276/S276_used.csv",
            pdb_dir="/home/hongtan/pretrain_single/data/dataset/276/pdb_s276",
            cache_dir="/home/hongtan/pretrain_single/data/dataset/276/cache",
            patch_size=128
        )
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)  # 80% 用于训练集
val_size = int(0.1 * dataset_size)    # 10% 用于验证集
test_size = dataset_size - train_size - val_size  # 剩下的用于测试集

# 使用 random_split 来划分数据集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
test_dataset = s276_dataset
# 创建 PyGDataLoader 对象
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=PaddingCollate(),num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True,collate_fn=PaddingCollate(),num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,collate_fn=PaddingCollate(),num_workers=4)

vae_model = Codebook(args).to(device)
vae_model.load_state_dict(torch.load("/home/hongtan/pretrain_single/weight/checkpoint/vae_model.ckpt", map_location=device))

model = DDG_RDE_Network(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

train_model(model, train_loader, val_loader,optimizer = optimizer, epochs=100, device= device)
best_model_path = "/home/hongtan/pretrain_single/weight/best_model.pth"
model.load_state_dict(torch.load(best_model_path))

# 在测试集上测试
test_model(model, test_loader, device=device)
