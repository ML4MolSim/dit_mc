import argparse
import glob
import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import datamol as dm
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType

from dit_mc.data_loader.utils import load_pkl, save_pkl
from dit_mc.generative_process.priors import compute_eigh_laplacian
from dit_mc.prepare_dataset import *


def check_smiles(smiles, smiles_mol):
    # filter mols rdkit can't intrinsically handle
    if smiles_mol is None:
        return False

    # skip conformers with fragments
    if '.' in smiles:
        return False
    
    # skip mols with atoms with more than 4 neighbors for now
    num_neighbors = [len(a.GetNeighbors()) for a in smiles_mol.GetAtoms()]
    if np.max(num_neighbors) > 4:
        return False
            
    return True


def get_mol_properties(mol):
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    atomic_numbers = np.array(get_atomic_numbers_from_mol(mol))
    bonds = np.array([
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        for bond in mol.GetBonds()
    ])

    return num_atoms, num_bonds, atomic_numbers, bonds


def check_conformer(conformer_mol, num_atoms, num_bonds, atomic_numbers, bonds):
    edge_case = False

    # mark mols with fragments as edge cases
    if len(Chem.GetMolFrags(conformer_mol)) > 1:
        edge_case = True

    # skip mols with atoms with more than 4 neighbors for now
    # this is in line with the MCF preprocessing 
    # here: https://github.com/apple/ml-mcf/blob/main/process_data.py
    num_neighbors = [len(a.GetNeighbors()) for a in conformer_mol.GetAtoms()]
    if np.max(num_neighbors) > 4:
        return "invalid"

    # mark mols that appear to be not a real conformer as edge cases
    # this ensures the same connectivity structure across all mols
    other_num_atoms, other_num_bonds, other_atomic_numbers, other_bonds = get_mol_properties(conformer_mol)
    if num_atoms != other_num_atoms:
        edge_case = True

    if num_bonds != other_num_bonds:
        edge_case = True

    if not edge_case:
        # being here means number of atoms and bonds are matching
        target = np.zeros(num_atoms)
        if not np.allclose(atomic_numbers - other_atomic_numbers, target):
            edge_case = True

        target = np.zeros((num_bonds, 2))
        if not np.allclose(bonds - other_bonds, target):
            edge_case = True

    if edge_case:
        return "edge_case"
    return "valid"


def my_filter_mols(mol_dict, max_confs=None):
    confs = mol_dict['conformers']
    smiles = mol_dict['smiles']

    # sort conformers by their Boltzmann weight
    confs = sorted(confs, key=lambda conf: -conf['boltzmannweight'])

    # get number of conformers
    num_confs = len(confs)
    if max_confs is not None:
        num_confs = max_confs

    mol = Chem.MolFromSmiles(smiles)

    # filter smiles mol
    if not check_smiles(smiles, mol):
        return []

    k = 0
    mols = []
    # we load the first conformer to get the properties
    # maybe we should rather load it directly from the smiles mol?
    mol = confs[0]['rd_mol']
    num_atoms, num_bonds, atomic_numbers, bonds = get_mol_properties(mol)
    
    for conf in confs:
        mol = conf['rd_mol']
        result = check_conformer(mol, num_atoms, num_bonds, atomic_numbers, bonds)
        edge_case = False
        
        if result == "invalid":
            continue
        elif result == "edge_case":
            edge_case = True

        mols.append({"mol": mol, "edge_case": edge_case})

        k += 1
        if k == num_confs:
            break

    return mols


# TODO: this is inefficient: first we construct the whole graph with edges
# Then we filter out the edges that are not bonds
# We should construct the graph with only the bonds
def cutoff_graph_to_bond_graph(g):
    """
        Make a bond graph from a cutoff graph.
    """
    edge_msk = g['edge_mask'] == 1
    senders = g["src_idx"][edge_msk]
    receivers = g["dst_idx"][edge_msk]
    edge_attr = g["edge_attr"][edge_msk]

    return {
        "senders": senders, 
        "receivers": receivers, 
        "edge_attr": edge_attr[:, :4] # don't include the MISC bit
    }


def rows_to_sample(graph_row, prior_graph_row, bond_graph, smiles, smiles_corrected, smiles_index, 
                   edge_case, chiral_nbr_index=None, chiral_tag=None, rdkit_pickle=None):
    if chiral_nbr_index is None:
        chiral_nbr_index = np.array([], dtype=np.int32)
    if chiral_tag is None:
        chiral_tag = np.array([], dtype=np.int8)
    if rdkit_pickle is None:
        rdkit_pickle = ""
    
    return {
        'graph_x1': graph_row['x1'].astype(np.float32),
        'graph_node_attr': graph_row['node_attr'].astype(np.uint8),
        'graph_atomic_numbers': graph_row['atomic_numbers'].astype(np.uint8),
        'graph_shortest_hops': graph_row['shortest_hops'].astype(np.uint16),
        'prior_node_attr': prior_graph_row['node_attr'].astype(np.float32),
        'prior_edge_attr': prior_graph_row['edge_attr'].astype(np.float32),
        'prior_senders': prior_graph_row['src_idx'].astype(np.uint8),
        'prior_receivers': prior_graph_row['dst_idx'].astype(np.uint8),
        'cond_edge_attr': bond_graph['edge_attr'].astype(np.uint8),
        'cond_senders': bond_graph['senders'].astype(np.uint8),
        'cond_receivers': bond_graph['receivers'].astype(np.uint8),
        'smiles': smiles,
        'smiles_corrected': smiles_corrected,
        'smiles_index': np.array(smiles_index).astype(np.int32),
        'edge_case': np.array(edge_case).astype(np.uint8),
        'chiral_nbr_index': chiral_nbr_index.astype(np.uint8).reshape(-1, 4),
        'chiral_tag': chiral_tag.astype(np.int8).reshape(-1),
        'rdkit_pickle': rdkit_pickle,
    }


def get_chiral_centers_from_mol(mol):
    """Only consider chiral atoms with 4 neighbors"""
    chiral_index = np.array(
        [
            i
            for i, atom in enumerate(mol.GetAtoms())
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=np.int32,
    ) # (n_chiral_centers,)

    chiral_nbr_index = np.array(
        [
            [n.GetIdx() for n in atom.GetNeighbors()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=np.int32,
    ).flatten() # (n_chiral_centers * 4,)

    chiral_tag = np.array(
        [
            chirality[atom.GetChiralTag()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=np.int8,
    ) # (n_chiral_centers,)

    return chiral_index, chiral_nbr_index, chiral_tag


def yield_extra_test_set(datadir, dataset):
    ground_truth_mols = load_pkl(os.path.join(datadir, 'geom', dataset, 'test_mols.pkl'))
    shortest_hops_cache = {}
    eig_val_cache, eig_vec_cache = {}, {}
    for smiles, mols in tqdm(ground_truth_mols.items()):
        mol = Chem.MolFromSmiles(smiles)
        if not check_smiles(smiles, mol):
            continue
            
        for mol in mols:
            smiles_ = dm.to_smiles(
                mol,
                canonical=False,
                explicit_hs=True,
                with_atom_indices=True,
                isomeric=True,
            )
    
            (
                graph_row, 
                shortest_hops_cache
            ) = get_graph_row_from_mol(mol, smiles_, dataset, shortest_hops_cache)

            (
                prior_graph_row, 
                eig_val_cache, 
                eig_vec_cache 
            ) = get_prior_graph_row_from_mol(mol, smiles_, eig_val_cache, eig_vec_cache)
            bond_graph = cutoff_graph_to_bond_graph(graph_row)

            _, chiral_nbr_index, chiral_tag = get_chiral_centers_from_mol(mol)

            # we set edge case to false as we don't check individual conformers here
            # we also don't have a smiles index from the dataset
            yield rows_to_sample(graph_row=graph_row, prior_graph_row=prior_graph_row, bond_graph=bond_graph, smiles=smiles,
                                smiles_corrected=smiles_, smiles_index=-1, edge_case=False, chiral_nbr_index=chiral_nbr_index,
                                chiral_tag=chiral_tag, rdkit_pickle=pickle.dumps(mol))


def yield_dataset(datadir, dataset, mode, max_confs):
    if mode == "test":
        assert max_confs is None, "max_confs should be None for test split"

    split_idx = {
        'train': 0,
        'val': 1,
        'test': 2,
    }[mode]

    split = np.load(
        os.path.join(datadir, 'geom', dataset, 'split0.npy'), 
        allow_pickle=True
    )[split_idx]
    
    all_files = sorted(
        glob.glob(os.path.join(datadir, 'rdkit_folder', dataset, '*.pickle')))
    all_files = [(i, f) for i, f in enumerate(all_files) if i in split]
    
    print("Start processing...")
    shortest_hops_cache = {}
    eig_val_cache, eig_vec_cache = {}, {}
    for (smiles_index, f_path) in tqdm(all_files):
        mol_dict = load_pkl(f_path)
        smiles = mol_dict['smiles']
        mols = my_filter_mols(mol_dict, max_confs=max_confs)
        for mol_desc in mols:
            mol = mol_desc["mol"]
            edge_case = mol_desc["edge_case"]

            # this is the corrected smiles string
            # it should be used for the eigenvalues / eigenvectors
            smiles_ = dm.to_smiles(
                mol,
                canonical=False,
                explicit_hs=True,
                with_atom_indices=True,
                isomeric=True,
            )

            (
                graph_row, 
                shortest_hops_cache
            ) = get_graph_row_from_mol(mol, smiles_, dataset, shortest_hops_cache)
            
            (
                prior_graph_row, 
                eig_val_cache, 
                eig_vec_cache 
            ) = get_prior_graph_row_from_mol(mol, smiles_, eig_val_cache, eig_vec_cache)
            bond_graph = cutoff_graph_to_bond_graph(graph_row)

            # store both the corrected and the original smiles string
            yield rows_to_sample(graph_row=graph_row, prior_graph_row=prior_graph_row, bond_graph=bond_graph, smiles=smiles,
                                smiles_corrected=smiles_, smiles_index=smiles_index, edge_case=edge_case)

# GRAPH ROW
    # graph_row = {
    #     'atomic_numbers': atomic_numbers,
    #     'shortest_hops': shortest_hops,
    #     'node_attr': node_attr,
    #     'x1': x1,
    #     'edge_mask': edge_mask,
    #     'edge_attr': edge_attr,
    #     'src_idx': src_idx,
    #     'dst_idx': dst_idx,
    #     'smiles': smiles,
    #     'n_node': n_node,
    #     'n_edge': n_edge,
    # }

# PRIOR GRAPH ROW
    # prior_graph_row = {
    #     'node_attr': D,
    #     'edge_attr': P.flatten(),
    #     'src_idx': src_idx,
    #     'dst_idx': dst_idx,
    #     'smiles': smiles,
    #     'n_node': n_node,
    #     'n_edge': n_edge,
    # }
