import os
import pandas as pd
import numpy as np
import json,pickle
import torch
from torch.utils.data import Dataset

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import one_to_index, index_to_one
from common_utils.transforms import get_transform
from common_utils.protein.parsers import parse_biopython_structure

import esm
from tqdm.auto import tqdm
from ipdb import set_trace
from multiprocessing import Pool, cpu_count



def process_entry(e):
    pdbcode = e['pdbcode']
    mutations = e['mutations']
    parser = PDBParser(QUIET=True)
    pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.lower()))
    model = parser.get_structure(None, pdb_path)[0]
    data, seq_map = parse_biopython_structure(model)

    aa_mut = data['aa'].clone()

    for mut in mutations:
        ch_rs_ic = (mut['resseq'])
        if ch_rs_ic > len(data['aa']):
            print(pdbcode)
            print(mutations)
            return None
        if aa_mut[ch_rs_ic] == one_to_index(mut['wt']):
            aa_mut[ch_rs_ic] = one_to_index(mut['mt'])
        else:
            print(mutations)
            print('fuck')
            return None

    data['aa_mut'] = aa_mut

    seq_wt = ''.join(index_to_one(aa) for aa in data['aa'])
    seq_mut = ''.join(index_to_one(aa) for aa in data['aa_mut'])

    return seq_wt, seq_mut

if __name__ == "__main__":

    with open('/home/hongtan/pretrain_single/thermompnn_dataset/dataset/fireprot/cache/entries.pkl', 'rb') as f:
        entries = pickle.load(f)

    protein_sequences = set()

    pdb_dir = '/home/hongtan/pretrain_single/thermompnn_dataset/dataset/fireprot/pdb'
    
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_entry, entries), total=len(entries), desc='pre_process'):
            if result is None:
                continue
            seq_wt, seq_mut = result
            protein_sequences.add(seq_wt)
            protein_sequences.add(seq_mut)

    output_dir = '/home/hongtan/pretrain_single/thermompnn_dataset/dataset/fireprot'
    os.makedirs(output_dir, exist_ok=True)
    protein_sequences_list = list(protein_sequences)
    json_path = os.path.join(output_dir, 'all_protein_sequences.json')
    with open(json_path, 'w') as json_file:
        json.dump(protein_sequences_list, json_file)
    
    print(f"Protein sequences have been saved to {json_path}")
