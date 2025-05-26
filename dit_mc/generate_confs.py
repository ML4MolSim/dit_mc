import argparse
import os
import hydra
import pickle
import numpy as np
import jax.numpy as jnp
import jax.random as random
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from copy import deepcopy
from rdkit.Geometry import Point3D
from rdkit.Chem.rdchem import ChiralType, Conformer

from dit_mc.data_loader.utils import save_pkl
from dit_mc.jraph_utils import compute_batch_statistics, get_number_of_nodes
from dit_mc.training.checkpoint import load_params_from_workdir
from dit_mc.training.utils import dynamically_batch_extended

chirality = {
  ChiralType.CHI_TETRAHEDRAL_CW: -1.0,
  ChiralType.CHI_TETRAHEDRAL_CCW: 1.0,
  ChiralType.CHI_UNSPECIFIED: 0,
  ChiralType.CHI_OTHER: 0,
}

def build_conformer(pos):
    if isinstance(pos, jnp.ndarray) or isinstance(pos, np.ndarray):
        pos = pos.tolist()

    conformer = Conformer()
    for i, atom_pos in enumerate(pos):
        conformer.SetAtomPosition(i, Point3D(*atom_pos))

    return conformer


def set_rdmol_positions(rdkit_mol, pos):
    mol = deepcopy(rdkit_mol)
    for conf_id in range(mol.GetNumConformers()):
        mol.RemoveConformer(conf_id)
    conformer = build_conformer(pos)
    mol.AddConformer(conformer)
    return mol


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
        dtype=np.float32,
    ) # (n_chiral_centers,)

    return chiral_index, chiral_nbr_index, chiral_tag


def signed_volume(local_coords):
    """
    Compute signed volume given ordered neighbor local coordinates

    :param local_coords: (n_tetrahedral_chiral_centers, 4, 3)
    :return: signed volume of each tetrahedral center
    (n_tetrahedral_chiral_centers,)
    """
    v1 = local_coords[:, 0] - local_coords[:, 3]
    v2 = local_coords[:, 1] - local_coords[:, 3]
    v3 = local_coords[:, 2] - local_coords[:, 3]
    cp = np.cross(v2, v3)
    vol = np.sum(v1 * cp, axis=-1)
    return np.sign(vol)


def switch_parity_of_pos(pos, chiral_nbr_index, chiral_tag):
    chiral_nbr_index = chiral_nbr_index.reshape(-1, 4)
    sv = signed_volume(pos[chiral_nbr_index])
    z_flip = sv * chiral_tag # (num_centers,)

    node_factor = -1. if (z_flip == -1.0).all() else 1.
    flip_mat = jnp.diag(jnp.array([1., 1., node_factor]))
    pos = jnp.matmul(pos, flip_mat.T)
    return pos


def generate_confs_main(workdir, batch_size=128, seed=-1, ckpt_dir_name="checkpoints", step=-1, use_correction=False):
    cfg = OmegaConf.load(os.path.join(workdir, ".hydra", "config.yaml"))
    
    if seed == -1:
        rng = random.PRNGKey(cfg.globals.seed)
    else:
        rng = random.PRNGKey(seed)

    print("Instantiate data module...")
    num_repeats = 2

    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    data_cfg = cfg_resolved["data_loader"]["data_cfg"]
    data_cfg["n_proc"] = 0 # this prevents the thread pool from being created
    data_module = hydra.utils.instantiate(cfg.data_loader, data_cfg=data_cfg)
    dataset, num_test_samples = data_module(split="test_small")
    dataset = dataset.repeat(num_repeats)

    print("Instantiate generative process...")
    process = hydra.utils.instantiate(cfg.generative_process)

    print("Load params...")
    params, step = load_params_from_workdir(
        workdir,
        ckpt_dir_name=ckpt_dir_name,
        step=step,
    )

    print("Start inference...")
    global_offset = 0
    predicted_mols = {}
    num_atoms = cfg.data_loader.num_atoms_mean

    batch = dynamically_batch_extended(
        dataset.as_numpy_iterator(),
        n_graph=batch_size,
        n_node=(batch_size - 1) * num_atoms + 1,
        n_edge=(batch_size - 1) * num_atoms * num_atoms + 1,
        n_edge_cond=(batch_size - 1) * num_atoms * 4 + 1,
        n_edge_prior=(batch_size - 1) * num_atoms * num_atoms + 1,
    )

    total_approx = ((num_test_samples * num_repeats) // (max(2, batch_size) - 1)) + 1
    for graph, graph_cond, graph_prior in tqdm(batch, total=total_approx):

        if get_number_of_nodes(graph) == get_number_of_nodes(graph_cond) == get_number_of_nodes(graph_prior):
            pass
        else:
            n_graph, n_graph_cond, n_graph_prior = get_number_of_nodes(graph), get_number_of_nodes(graph_cond), get_number_of_nodes(graph_prior)
            raise RuntimeError(
                f'Number of nodes unequal after batching. '
                f'Received {n_graph=}, {n_graph_cond=} and {n_graph_prior=}.'
            )

        smiles_globals = graph.globals['smiles']
        rdkit_pickle_globals = graph.globals['rdkit_pickle']
        del graph.globals['smiles']
        del graph.globals['rdkit_pickle']

        rng_sample, rng = random.split(rng)
        sample = process.sample(params, graph, graph_prior, graph_cond, rng_sample)

        batch = compute_batch_statistics(sample)

        num_graphs_in_batch = sum(batch.graph_mask)
        for idx in range(num_graphs_in_batch):
            smiles = smiles_globals[idx].decode('utf-8')
            mol = pickle.loads(rdkit_pickle_globals[idx])
            positions = sample.nodes["positions"][batch.batch_segments == idx]

            if use_correction:
                _, chiral_nbr_index, chiral_tag = get_chiral_centers_from_mol(
                    mol
                )

                positions = switch_parity_of_pos(
                    positions, chiral_nbr_index, chiral_tag
                )

            if smiles not in predicted_mols:
                predicted_mols[smiles] = []

            predicted_mol = set_rdmol_positions(mol, positions)
            predicted_mols[smiles].append(predicted_mol)

        global_offset += num_graphs_in_batch

    print("Save results...")
    if seed == -1:
        used_cc = "CC" if use_correction else "plain"
        save_pkl(os.path.join(workdir, f"pred_mols_epoch{step}_{used_cc}.pkl"), predicted_mols)
    else:
        used_cc = "CC" if use_correction else "plain"
        save_pkl(os.path.join(workdir, f"pred_mols_epoch{step}_seed{seed}_{used_cc}.pkl"), predicted_mols)

    print("Done.")

    # return the actual step used to load the checkpoint
    return step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", 
        type=str, 
        help="Path of the directory containing the model parameters.", 
        required=True
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=128, 
        help="Batch size. Default 128."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=-1, 
        help="Seed. Default -1 (Use cfg.globals.seed)."
    )
    parser.add_argument(
        "--ckpt_dir_name", 
        type=str, 
        default='last_checkpoint', 
        help="Name of the checkpoint directory.", 
    )
    parser.add_argument(
        "--step", 
        type=int, 
        default=-1,
        help="Checkpoint step (default: last available step).",
    )
    parser.add_argument(
        "--use_correction",
        action="store_true",
        default=True,
        help="Use chirality correction for the predictions.",
    )
    args = parser.parse_args()
    
    workdir = args.workdir
    batch_size = args.batch_size
    seed = args.seed
    ckpt_dir_name = args.ckpt_dir_name
    step = args.step
    use_correction = args.use_correction

    generate_confs_main(
        workdir=workdir,
        batch_size=batch_size,
        seed=seed,
        ckpt_dir_name=ckpt_dir_name,
        step=step,
        use_correction=use_correction
    )
