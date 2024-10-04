import os
import copy
import math
import random
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import one_to_index, index_to_one
from common_utils.transforms import get_transform
from common_utils.protein.parsers import parse_biopython_structure
import json
from ipdb import set_trace


with open('/home/hongtan/pretrain_single/dataset/final_dataset/protein_embeddings.pkl', 'rb') as f:
    esm2_embeddings = pickle.load(f)

class SkempiDataset(Dataset):

    def __init__(self, csv_path, pdb_dir, cache_dir, patch_size=128, reset=False):
        super().__init__()
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.transform = get_transform([{'type': 'select_atom', 'resolution': 'backbone+CB'}, {'type': 'selected_region_fixed_size_patch', 'select_attr': 'mut_flag', 'patch_size': patch_size}])

        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, 'structures.pkl')
        self.structures = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        # 使用全部数据
        self.entries = self.entries_full
        
    def _preprocess_entries(self):
        entries = load_skempi_entries(self.csv_path, self.pdb_dir)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        return entries

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures()
        else:
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)

    def _preprocess_structures(self):
        structures = {}
        pdbcodes = list(set([e['pdbcode'] for e in self.entries_full]))
        for pdbcode in tqdm(pdbcodes, desc='Structures'):
            parser = PDBParser(QUIET=True)
            pdb_path = os.path.join(self.pdb_dir, '{}.pdb'.format(pdbcode))
            model = parser.get_structure(None, pdb_path)[0]
            data, seq_map = parse_biopython_structure(model)
            structures[pdbcode] = (data, seq_map)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        data, seq_map = copy.deepcopy(self.structures[entry['pdbcode']])
        if data is None or seq_map is None:
            print(f"Skipping entry {entry['mutations']}: structure data not found.")
            return None
    
        data['ddG'] = entry['ddG']
        aa_mut = data['aa'].clone()

        for mut in entry['mutations']:
            ch_rs_ic = (mut['chain'], mut['resseq'])
            if ch_rs_ic not in seq_map:
                print(f"Skipping mutation {mut['wt']}->{mut['mt']} at {ch_rs_ic}: not found in seq_map")
                continue
            if aa_mut[seq_map[ch_rs_ic]] == one_to_index(mut['wt']):
                aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
            else:
                print(f"Skipping mutation {mut['wt']}->{mut['mt']} at {ch_rs_ic}: not found in seq_map")
                continue


        data['aa_mut'] = aa_mut
        data['mut_flag'] = (data['aa'] != data['aa_mut'])

        seq_wt = ''.join(index_to_one(aa) for aa in data['aa'])
        seq_mut = ''.join(index_to_one(aa) for aa in data['aa_mut'])

        esm_embeddings_wt = torch.tensor(esm2_embeddings[seq_wt])
        esm_embeddings_mut = torch.tensor(esm2_embeddings[seq_mut])

        
        data['esm_embeddings_wt'] = esm_embeddings_wt
        data['esm_embeddings_mut'] = esm_embeddings_mut
        
        data.pop('resseq', None)
        if self.transform is not None:
            data = self.transform(data)
            if data is None:  # 检查 transform 是否返回 None
                print(f"Skipping entry {index} after transform.")
                return None

        return data

def load_skempi_entries(csv_path, pdb_dir, block_list={'1KBH'}):
    df = pd.read_csv(csv_path, sep=',')

    def _parse_mut(mut_name):
        return {'wt': mut_name[0], 'mt': mut_name[-1], 'chain': mut_name[1],'resseq': str(mut_name[2:-1])}
    entries = []
    for i, row in df.iterrows():
        pdbcode = row['pdbcode'].lower()
        if pdbcode in block_list:
            continue
        if not os.path.exists(os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.lower()))):
            continue
        if not np.isfinite(row['ddg']):
            continue

        mutations = list(map(_parse_mut, row['mutation'].split(',')))
        entry = {
            'id': i,
            'pdbcode': pdbcode,
            'mutations': mutations,
            'num_muts': len(mutations),
            'ddG': np.float32(row['ddg']),
        }
        entries.append(entry)

    return entries

