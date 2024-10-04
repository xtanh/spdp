import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from easydict import EasyDict

from .constants import (AA, max_num_heavyatoms,
                        restype_to_heavyatom_names, 
                        BBHeavyAtom)
from .icoord import get_chi_angles, get_backbone_torsions


def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    bfactor_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.float)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True
            bfactor_heavyatom[idx] = res[atom_name].get_bfactor()
    return pos_heavyatom, mask_heavyatom, bfactor_heavyatom


def parse_pdb(path, model_id, unknown_threshold=1.0):
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)


def parse_mmcif_assembly(path, model_id, assembly_id=0, unknown_threshold=1.0):
    parser = MMCIFParser()
    structure = parser.get_structure(None, path)
    mmcif_dict = parser._mmcif_dict
    if '_pdbx_struct_assembly_gen.asym_id_list' not in mmcif_dict:
        return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)
    else:
        assemblies = [tuple(chains.split(',')) for chains in mmcif_dict['_pdbx_struct_assembly_gen.asym_id_list']]
        label_to_auth = {}
        for label_asym_id, auth_asym_id in zip(mmcif_dict['_atom_site.label_asym_id'], mmcif_dict['_atom_site.auth_asym_id']):
            label_to_auth[label_asym_id] = auth_asym_id
        model_real = list({structure[model_id][label_to_auth[ch]] for ch in assemblies[assembly_id]})
        return parse_biopython_structure(model_real)


def parse_biopython_structure(entity, unknown_threshold=1.0):
    chains = Selection.unfold_entities(entity, 'C')
    chains.sort(key=lambda c: c.get_id())
    
    data = EasyDict({
        'chain_id': [], 'chain_nb': [],
        'resseq': [], 'res_nb': [],
        'aa': [],
        'pos_heavyatom': [], 'mask_heavyatom': [],
        'phi': [], 'phi_mask': [],
        'psi': [], 'psi_mask': [],
        'omega': [], 'omega_mask': [],
        'chi': [], 'chi_mask': [],
    })
    tensor_types = {
        'chain_nb': torch.LongTensor,
        'resseq': torch.LongTensor,
        'res_nb': torch.LongTensor,
        'aa': torch.LongTensor,
        'pos_heavyatom': torch.stack,
        'mask_heavyatom': torch.stack,

        'phi': torch.FloatTensor,
        'phi_mask': torch.BoolTensor,
        'psi': torch.FloatTensor,
        'psi_mask': torch.BoolTensor,
        'omega': torch.FloatTensor,
        'omega_mask': torch.BoolTensor,

        'chi': torch.stack,
        'chi_mask': torch.stack,
    }

    count_aa, count_unk = 0, 0

    for i, chain in enumerate(chains):
        chain.atom_to_internal_coordinates()
        seq_this = 0   # Renumbering residues

        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode

        for _, res in enumerate(residues):
            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            #if not (res.has_id('CA') and res.has_id('C') and res.has_id('N') and res.has_id('O')): continue

            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            # Chain info
            data.chain_id.append(chain.get_id())
            data.chain_nb.append(i)

            # Residue types
            data.aa.append(restype)

            # Heavy atoms
            pos_heavyatom, mask_heavyatom, _ = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)

            # Backbone torsions
            phi, psi, omega = get_backbone_torsions(res)
            if phi is None:
                data.phi.append(0.0)
                data.phi_mask.append(False)
            else:
                data.phi.append(phi)
                data.phi_mask.append(True)
            if psi is None:
                data.psi.append(0.0)
                data.psi_mask.append(False)
            else:
                data.psi.append(psi)
                data.psi_mask.append(True)
            if omega is None:
                data.omega.append(0.0)
                data.omega_mask.append(False)
            else:
                data.omega.append(omega)
                data.omega_mask.append(True)

            # Chi
            chi, _, chi_mask, _ = get_chi_angles(restype, res)
            data.chi.append(chi)
            data.chi_mask.append(chi_mask)

            # Sequential number
            resseq = res.get_id()[1]  # Get the residue sequence number
            icode = res.get_id()[2].strip()  # Get the insertion code (icode) and strip any whitespace

            # Create a unique key for `resseq_this` by combining `resseq` and `icode` if `icode` is not empty
            if icode:
                resseq_this = f"{resseq}{icode}"
            else:
                resseq_this = str(resseq)
                
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = int(resseq) - int(data.resseq[-1].strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.res_nb.append(seq_this)

    if len(data.aa) == 0:
        return None, None

    if (count_unk / count_aa) >= unknown_threshold:
        return None, None

    seq_map = {}
    for i, (chain_id, resseq) in enumerate(zip(data.chain_id, data.resseq)):
        seq_map[(chain_id, resseq)] = i

    for key, convert_fn in tensor_types.items():
        if key == 'resseq':
            continue
        data[key] = convert_fn(data[key])
    return data, seq_map