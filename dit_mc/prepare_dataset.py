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
from rdkit.Chem import rdmolops

from dit_mc.data_loader.utils import load_pkl, save_pkl
from dit_mc.generative_process.priors import compute_eigh_laplacian

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

atomic_types = {
    'qm9': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'qm9_ablation': {
        'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7,
        'Mg': 8, 'Al': 9, 'Si': 10, 'P': 11, 'S': 12, 'Cl': 13, 'K': 14,
        'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20, 'Ga': 21,
        'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28,
        'I': 29, 'Gd': 30, 'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34
    },
    'drugs': {
        'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7,
        'Mg': 8, 'Al': 9, 'Si': 10, 'P': 11, 'S': 12, 'Cl': 13, 'K': 14,
        'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20, 'Ga': 21,
        'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28,
        'I': 29, 'Gd': 30, 'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34
    },
    'xl': {
        'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7,
        'Mg': 8, 'Al': 9, 'Si': 10, 'P': 11, 'S': 12, 'Cl': 13, 'K': 14,
        'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20, 'Ga': 21,
        'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28,
        'I': 29, 'Gd': 30, 'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34
    }
}


chirality = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0
}


bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


def one_hot_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for misc values.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def get_positions_from_mol(mol):
    return mol.GetConformer().GetPositions()


def get_atomic_numbers_from_mol(mol):
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return np.array(atomic_numbers, dtype=np.int8)


def get_node_attr_from_mol(mol, dataset):
    all_node_attr = []
    ring = mol.GetRingInfo()
    types = list(atomic_types[dataset].keys())
    for i, atom in enumerate(mol.GetAtoms()):
        node_attr = []
        node_attr.extend(one_hot_encoding(atom.GetChiralTag(), [
            ChiralType.CHI_TETRAHEDRAL_CW,
            ChiralType.CHI_TETRAHEDRAL_CCW,
            ChiralType.CHI_UNSPECIFIED,
            ChiralType.CHI_OTHER,
        ]))
        node_attr.extend(one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
        node_attr.extend(one_hot_encoding(atom.GetNumRadicalElectrons(), [0, 1, 2, 3, 4]))
        node_attr.extend(one_hot_encoding(atom.GetSymbol(), types)[:-1])
        node_attr.append(int(atom.GetIsAromatic()))
        node_attr.extend(one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4]))
        node_attr.extend(one_hot_encoding(atom.GetHybridization(), [
                                Chem.rdchem.HybridizationType.SP,
                                Chem.rdchem.HybridizationType.SP2,
                                Chem.rdchem.HybridizationType.SP3,
                                Chem.rdchem.HybridizationType.SP3D,
                                Chem.rdchem.HybridizationType.SP3D2]))
        node_attr.extend(one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4]))
        node_attr.extend(one_hot_encoding(atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]))
        node_attr.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                                int(ring.IsAtomInRingOfSize(i, 4)),
                                int(ring.IsAtomInRingOfSize(i, 5)),
                                int(ring.IsAtomInRingOfSize(i, 6)),
                                int(ring.IsAtomInRingOfSize(i, 7)),
                                int(ring.IsAtomInRingOfSize(i, 8))])
        node_attr.extend(one_hot_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))
        all_node_attr.append(node_attr)

    return np.array(all_node_attr, dtype=np.int8)


def get_src_and_dst_idx_from_mol(mol):
    # for now we will use a fully connected graph w/o self interactions
    num_atoms = mol.GetNumAtoms()
    mask = ~np.eye(num_atoms, dtype=np.bool_) # get rid of self interactions
    keep_edges = np.where(mask)
    src_idx = keep_edges[0].astype(np.uint8)
    dst_idx = keep_edges[1].astype(np.uint8)

    # initialize edge information
    bonds_values = list(bonds.values())
    edge_mask = np.zeros((num_atoms, num_atoms), dtype=np.int8)
    edge_attr = np.zeros((num_atoms, num_atoms, len(bonds_values) + 1), dtype=np.int8)
    for i in range(num_atoms):
      for j in range(num_atoms):
          edge_feat = one_hot_encoding(-100, bonds_values) # force misc value
          edge_attr[i, j, :] = edge_feat

    # set correct edge information for covalent bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        edge_mask[i, j] = 1
        edge_mask[j, i] = 1

        edge_feat = one_hot_encoding(bonds[bond.GetBondType()], bonds_values)
        edge_attr[i, j, :] = edge_feat
        edge_attr[j, i, :] = edge_feat

    # get rid of self interactions
    edge_mask = edge_mask[mask]
    edge_attr = edge_attr[mask]

    return src_idx, dst_idx, edge_mask, edge_attr


def filter_mols(mol_dict, max_confs=None):
    confs = mol_dict['conformers']
    smiles = mol_dict['smiles']

    # get number of conformers
    num_confs = len(confs)
    if max_confs is not None:
        num_confs = max_confs

    # filter mols rdkit can't intrinsically handle
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    # skip conformers with fragments
    if '.' in smiles:
        return []

    k = 0
    mols = []
    for conf in confs:
        mol = conf['rd_mol']

        # skip mols with atoms with more than 4 neighbors for now
        num_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        if np.max(num_neighbors) > 4:
            continue

        mols.append(mol)

        k += 1
        if k == num_confs:
            break

    return mols


def get_graph_row_from_mol(mol, smiles, dataset, shortest_hops_cache):

    num_atoms = mol.GetNumAtoms()

    x1 = get_positions_from_mol(mol)
    node_attr = get_node_attr_from_mol(mol, dataset)
    atomic_numbers = get_atomic_numbers_from_mol(mol)
    src_idx, dst_idx, edge_mask, edge_attr = get_src_and_dst_idx_from_mol(mol)

    if smiles in shortest_hops_cache:
        shortest_hops = shortest_hops_cache[smiles]
    else:
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        shortest_path_result, _ = algos.floyd_warshall(adj_matrix)
        mask = ~np.eye(num_atoms, dtype=np.bool_) # get rid of self interactions
        shortest_hops = shortest_path_result[mask]
        
        shortest_hops_cache[smiles] = shortest_hops

    n_node = np.array([num_atoms], dtype=np.uint8)
    n_edge = np.array([len(src_idx)], dtype=np.int32)

    graph_row = {
        'atomic_numbers': atomic_numbers,
        'node_attr': node_attr,
        'x1': x1,
        'edge_mask': edge_mask,
        'edge_attr': edge_attr,
        'shortest_hops': shortest_hops,
        'src_idx': src_idx,
        'dst_idx': dst_idx,
        'smiles': smiles,
        'n_node': n_node,
        'n_edge': n_edge,
    }

    return graph_row, shortest_hops_cache


def get_prior_graph_row_from_mol(mol, smiles, eig_val_cache, eig_vec_cache, threshold=0.0001):

    num_atoms = mol.GetNumAtoms()

    try:
        # try to get number of fragments explitictly
        num_rdkit_components = len(rdmolops.GetMolFrags(mol))
    except:
        # use eigenvalue threshold to get number of fragments
        num_rdkit_components = None

    if smiles in eig_val_cache and smiles in eig_vec_cache:
        D = eig_val_cache[smiles]
        P = eig_vec_cache[smiles]
    else:
        src_idx, dst_idx = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src_idx += [i, j]
            dst_idx += [j, i]
        src_idx = np.array(src_idx, dtype=np.uint8)
        dst_idx = np.array(dst_idx, dtype=np.uint8)

        D, P = compute_eigh_laplacian(src_idx, dst_idx, num_nodes=num_atoms)
        explained_variance_ratio = D / D.sum()
        D = 1 / jnp.sqrt(D)

        # avoid large scalars (e.g. for conformers with disconnected components)
        if num_rdkit_components is not None:
            D = D.at[:num_rdkit_components].set(0)
        else:
            print("Error in rdkit fragments. Using manual eigenvalue threshold.")
            # manually set the zero eigenvalues to 0
            D = jax.numpy.where(explained_variance_ratio < threshold, 0., D)

        if jnp.any(D > 1 / jnp.sqrt(threshold)):
            print("---\nWarning: small eigenvalues detected. This might lead to numerical instability.\n----")

        eig_val_cache[smiles] = D
        eig_vec_cache[smiles] = P

    x, y = jnp.arange(num_atoms), jnp.arange(num_atoms)
    src_idx, dst_idx = jnp.meshgrid(x, y)
    src_idx = src_idx.flatten().astype(jnp.uint8)
    dst_idx = dst_idx.flatten().astype(jnp.uint8)

    n_node = np.array([num_atoms], dtype=np.uint8)
    n_edge = np.array([len(src_idx)], dtype=np.int32)

    prior_graph_row = {
        'node_attr': D,
        'edge_attr': P.flatten(),
        'src_idx': src_idx,
        'dst_idx': dst_idx,
        'smiles': smiles,
        'n_node': n_node,
        'n_edge': n_edge,
    }

    return prior_graph_row, eig_val_cache, eig_vec_cache


def prepare_train_or_val_dataset(datadir, dataset, mode, max_confs):
    split = np.load(
        os.path.join(datadir, 'geom', dataset, 'split0.npy'), 
        allow_pickle=True
    )[0 if mode == 'train' else 1]
    
    all_files = sorted(
        glob.glob(os.path.join(datadir, 'rdkit_folder', dataset, '*.pickle')))
    all_files = [f for i, f in enumerate(all_files) if i in split]
    
    print("Start processing...")
    graph_rows, prior_graph_rows = [], []
    shortest_hops_cache = {}
    eig_val_cache, eig_vec_cache = {}, {}
    for f_path in tqdm(all_files):
        mol_dict = load_pkl(f_path)
        smiles = mol_dict['smiles']
        mols = filter_mols(mol_dict, max_confs=max_confs)
        for mol in mols:
            
            (
                graph_row, 
                shortest_hops_cache
            ) = get_graph_row_from_mol(mol, smiles, dataset, shortest_hops_cache)
            graph_rows.append(graph_row)
            
            (
                prior_graph_row, 
                eig_val_cache, 
                eig_vec_cache 
            ) = get_prior_graph_row_from_mol(mol, smiles, eig_val_cache, eig_vec_cache)
            prior_graph_rows.append(prior_graph_row)
    
    # unfortunately this may take a while
    print("Save to disk...")
    pl_graphs = pl.DataFrame(graph_rows)
    pl_graphs = pl_graphs.with_columns(
        pl.col("atomic_numbers").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Int8)),
        pl.col("node_attr").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.List(pl.Int8))),
        pl.col("x1").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.List(pl.Float64))),
        pl.col("edge_attr").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.List(pl.Int8))),
        pl.col("edge_mask").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Int8)),
        pl.col("src_idx").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("dst_idx").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("smiles").map_elements(
            lambda x: x, return_dtype=pl.String),
        pl.col("n_node").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("n_edge").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Int32)),
    )

    pl_graphs.write_parquet(
        os.path.join(datadir, 'geom', dataset, f'{mode}_graphs.parquet'))
    
    pl_prior_graphs = pl.DataFrame(prior_graph_rows)
    pl_prior_graphs = pl_prior_graphs.with_columns(
        pl.col("node_attr").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Float64)),
        pl.col("edge_attr").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Float64)),
        pl.col("src_idx").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("dst_idx").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("smiles").map_elements(
            lambda x: x, return_dtype=pl.String),
        pl.col("n_node").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("n_edge").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Int32)),
    )    

    pl_prior_graphs.write_parquet(
        os.path.join(datadir, 'geom', dataset, f'{mode}_prior_graphs.parquet'))


def prepare_test_dataset(datadir, dataset, confs_per_mol=2):
    ground_truth_mols = load_pkl(os.path.join(datadir, 'geom', dataset, 'test_mols.pkl'))

    print("Start processing...")
    test_smiles = []
    graph_rows, prior_graph_rows = [], []
    shortest_hops_cache = {}
    eig_val_cache, eig_vec_cache = {}, {}
    for smiles, mols in tqdm(ground_truth_mols.items()):
        mol = Chem.MolFromSmiles(smiles)

        # skip mols rdkit can't intrinsically handle
        if mol is None:
            print(f"Skip {smiles} due to rdkit error.")
            continue

        # skip conformers with fragments
        if '.' in smiles:
            print(f"Skip {smiles} due to disconnected component.")
            continue

        # skip mols with atoms with more than 4 neighbors for now
        num_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        if np.max(num_neighbors) > 4:
            print(f"Skip {smiles} due to large degree.")
            continue
            
        # create `confs_per_mol` graphs per mol
        for mol in mols:
            smiles_ = dm.to_smiles(
                mol,
                canonical=False,
                explicit_hs=True,
                with_atom_indices=True,
                isomeric=True,
            )
    
            test_smiles.extend([smiles] * confs_per_mol)
            (
                graph_row, 
                shortest_hops_cache
            ) = get_graph_row_from_mol(mol, smiles_, dataset, shortest_hops_cache)
            graph_rows.extend([graph_row] * confs_per_mol)

            (
                prior_graph_row, 
                eig_val_cache, 
                eig_vec_cache 
            ) = get_prior_graph_row_from_mol(mol, smiles_, eig_val_cache, eig_vec_cache)
            prior_graph_rows.extend([prior_graph_row] * confs_per_mol)

    print("Save to disk...")
    save_pkl(os.path.join(datadir, 'geom', dataset, 'test_smiles.pkl'), test_smiles)

    pl_graphs = pl.DataFrame(graph_rows)
    pl_graphs = pl_graphs.with_columns(
        pl.col("atomic_numbers").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Int8)),
        pl.col("node_attr").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.List(pl.Int8))),
        pl.col("x1").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.List(pl.Float64))),
        pl.col("edge_attr").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.List(pl.Int8))),
        pl.col("edge_mask").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Int8)),
        pl.col("src_idx").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("dst_idx").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("smiles").map_elements(
            lambda x: x, return_dtype=pl.String),
        pl.col("n_node").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("n_edge").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Int32)),
    )

    pl_graphs.write_parquet(
        os.path.join(datadir, 'geom', dataset, 'test_graphs.parquet'))

    pl_prior_graphs = pl.DataFrame(prior_graph_rows)
    pl_prior_graphs = pl_prior_graphs.with_columns(
        pl.col("node_attr").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Float64)),
        pl.col("edge_attr").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Float64)),
        pl.col("src_idx").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("dst_idx").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("smiles").map_elements(
            lambda x: x, return_dtype=pl.String),
        pl.col("n_node").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.UInt8)),
        pl.col("n_edge").map_elements(
            lambda x: x.tolist(), return_dtype=pl.List(pl.Int32)),
    )

    pl_prior_graphs.write_parquet(
        os.path.join(datadir, 'geom', dataset, 'test_prior_graphs.parquet'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", 
        type=str, 
        help="Path of the data directory.", 
        required=True
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="Name of the dataset. Please choose dataset from ['qm9', 'drugs'].", 
        required=True
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        help="Mode of the dataset. Please choose mode from ['train', 'val', 'test'].", 
        required=True
    )
    parser.add_argument(
        "--max_confs", 
        type=int, 
        default=-1, 
        help="Maximum number of conformers for a single smiles. Default -1 (no limit)."
    )
    args = parser.parse_args()

    assert args.dataset in ['qm9', 'drugs'], "Please choose dataset from ['qm9', 'drugs']"
    
    datadir = args.datadir
    dataset = args.dataset
    mode = args.mode
    max_confs = args.max_confs

    if max_confs == -1:
        max_confs = None

    if mode == 'train' or mode == 'val':
        prepare_train_or_val_dataset(datadir, dataset, mode, max_confs)
    elif mode == 'test':
        prepare_test_dataset(datadir, dataset)
    else:
        raise ValueError(f"Unknown mode: {mode}. Please choose mode from ['train', 'val', 'test']")
