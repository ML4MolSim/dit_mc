import pickle
from typing import Optional, Tuple
import polars as pl
import jax
import jraph
import numpy as np
import jax.numpy as jnp
import tensorflow as tf

from tqdm.auto import tqdm
from dit_mc.training.utils import kabsch_algorithm
from dit_mc.generative_process.priors import get_mol_from_smiles


BOND_TYPE_LIST = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"]


def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    

def save_pkl(file_path: str, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except Exception as e:
        return len(l) - 1
    

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    
    return safe_index(
        BOND_TYPE_LIST, str(bond.GetBondType())
    )


def compute_src_and_dst_index(
    mol, 
    reverse: Optional[bool] = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    src_idx, dst_idx, bond_type = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src_idx.append(i)
        dst_idx.append(j)
        bond_type.append(bond_to_feature_vector(bond))

        if reverse:
            src_idx.append(j)
            dst_idx.append(i)
            bond_type.append(bond_to_feature_vector(bond))

    if len(src_idx) == 0:
        src_idx = np.empty((0,), dtype=np.int32)
        dst_idx = np.empty((0,), dtype=np.int32)
        bond_type = np.empty((0,), dtype=np.int32)
        return src_idx, dst_idx

    src_idx = np.array(src_idx, dtype=np.int32)
    dst_idx = np.array(dst_idx, dtype=np.int32)
    bond_type = np.array(bond_type, dtype=np.int32)

    return src_idx, dst_idx, bond_type


def compute_dst_and_src_index_np(
        positions,
        cutoff: float,
        calculate_bonds: bool = False,
        bond_cutoff: float = None
):
    """Computes an edge list from atom positions and a fixed cutoff radius."""

    if calculate_bonds is True:
        assert bond_cutoff is not None

    num_atoms = positions.shape[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = np.linalg.norm(displacements, axis=-1)
    mask = ~np.eye(num_atoms, dtype=np.bool_)  # get rid of self interactions
    keep_edges = np.where((distances < cutoff) & mask)
    dst = keep_edges[0].astype(np.int32)
    src = keep_edges[1].astype(np.int32)

    if calculate_bonds is True:
        bonds = (np.linalg.norm(positions[dst] - positions[src], axis=-1) < bond_cutoff).astype(int) + 1
    else:
        bonds = None
    return dst, src, bonds


def get_conformer_graphs_from_trajectory_npz(
        data,
        num_samples: int,
        cutoff: float = 5.,
        shuffle: bool = True,
        calculate_bonds: bool = False,
        bond_cutoff: float = None,
        load_atomic_representations_bool: bool = False,
        seed: int = 0,
        center_structures = True
):
    np_rng = np.random.default_rng(seed)

    num_data = len(data['R'])
    atomic_numbers = data['z']

    # velocities are estimated as pos_i - pos_{i-1} so start index at 1.
    indx = np.arange(1, num_data)

    if shuffle:
        indx = np_rng.permutation(indx)

    graphs = []
    for n, k in enumerate(indx):
        if n >= num_samples:
            break

        pos_k = data['R'][k]

        if center_structures:
            pos_k = pos_k - pos_k.mean(axis=0, keepdims=True)

        src_idx, dst_idx, bonds = compute_dst_and_src_index_np(
            cutoff=cutoff,
            positions=pos_k,
            calculate_bonds=calculate_bonds,
            bond_cutoff=bond_cutoff,
        )

        g = jraph.GraphsTuple(
            nodes={
                'atomic_numbers': atomic_numbers,
                'positions': pos_k,
                'x1': pos_k,
            },
            edges={
                'bonds': bonds
            },
            globals={
                't': np.array([0.])
            },
            senders=src_idx,
            receivers=dst_idx,
            n_node=np.array([pos_k.shape[-2]]),
            n_edge=np.array([len(src_idx)]),
        )

        if load_atomic_representations_bool is True:
            g.nodes['atomic_representations'] = data['atomic_representations'][k]

        graphs.append(g)

    return graphs


def get_graphs_from_trajectory_npz(
        data,
        delta_frames: int,
        num_samples: int,
        cutoff: float = 5.,
        shuffle: bool = True,
        calculate_bonds: bool = False,
        bond_cutoff: float = None,
        load_atomic_representations_bool: bool = False,
        align_geometries_bool: bool = True,
        seed: int = 0,
        center_structures = True
):
    np_rng = np.random.default_rng(seed)

    num_data = len(data['R'])
    atomic_numbers = data['z']

    # velocities are estimated as pos_i - pos_{i-1} so start index at 1.
    indx = np.arange(1, num_data - delta_frames)

    if shuffle:
        indx = np_rng.permutation(indx)

    graphs = []
    for n, k in enumerate(indx):
        if n < num_samples:
            pos_k = data['R'][k]
            pos_kp1 = data['R'][k + delta_frames]
            pos_km1 = data['R'][k - 1]

            if align_geometries_bool is True:
                # aligns current coordinates (delta_frames in the future) with reference (t=0) coordinates
                pos_kp1 = kabsch_algorithm(
                    pos_kp1,  # current
                    pos_k  # reference
                )
                # aligns current coordinates (minus one frame) with reference (t=0) coordinates
                # only important if velocities are used such that they point in the appropriate direction
                pos_km1 = kabsch_algorithm(
                    pos_km1,  # current
                    pos_k  # reference
                )

            src_idx, dst_idx, bonds = compute_dst_and_src_index_np(
                cutoff=cutoff,
                positions=pos_k,
                calculate_bonds=calculate_bonds,
                bond_cutoff=bond_cutoff,
            )

            if center_structures:
                pos_k_center = pos_k - pos_k.mean(axis=0, keepdims=True)
                pos_kp1_center = pos_kp1 - pos_kp1.mean(axis=0, keepdims=True)
            else:
                pos_k_center = pos_k
                pos_kp1_center = pos_kp1

            g = jraph.GraphsTuple(
                nodes={
                    'atomic_numbers': atomic_numbers,
                    'positions': pos_k_center,
                    'velocities': pos_k - pos_km1,
                    'z': pos_kp1 - pos_k,
                    'x0': pos_k_center,
                    'x1': pos_kp1_center
                },
                edges={
                    'bonds': bonds
                },
                globals={
                    't': np.array([0.])
                },
                senders=src_idx,
                receivers=dst_idx,
                n_node=np.array([pos_k.shape[-2]]),
                n_edge=np.array([len(src_idx)]),
            )

            if load_atomic_representations_bool is True:
                g.nodes['atomic_representations'] = data['atomic_representations'][k]

            graphs.append(g)

        else:

            break

    return graphs


def get_graph_from_row(row):
    graph = jraph.GraphsTuple(
        nodes={
            'x1': np.array(row['x1']),
            'node_attr': np.array(row['node_attr']),
            'atomic_numbers': np.array(row['atomic_numbers']),
        },
        edges={
            'edge_mask': np.array(row['edge_mask']),
            'edge_attr': np.array(row['edge_attr']),
        },
        globals=None,
        senders=np.array(row['src_idx']),
        receivers=np.array(row['dst_idx']),
        n_node=np.array(row['n_node']),
        n_edge=np.array(row['n_edge']),
    )
    return graph


def get_prior_graph_from_row(row):
    prior_graph = jraph.GraphsTuple(
        nodes={
            'node_attr': np.array(row['node_attr']),
        },
        edges={
            'edge_attr': np.array(row['edge_attr']),
        },
        globals=None,
        senders=np.array(row['src_idx']),
        receivers=np.array(row['dst_idx']),
        n_node=np.array(row['n_node']),
        n_edge=np.array(row['n_edge']),
    )
    return prior_graph


def cutoff_graph_to_bond_graph_old(g):
    """
        Make a bond graph from a cutoff graph.
    """
    edge_msk = g.edges['edge_mask'] == 1
    
    senders = g.senders[edge_msk]
    receivers = g.receivers[edge_msk]

    edges_dict = jax.tree_util.tree_map(lambda x: x[edge_msk], g.edges)
    nodes_dict = jax.tree_util.tree_map(lambda x: x, g.nodes)

    n_edge = np.array([len(senders)])
    n_node = g.n_node
    globals = None
    
    
    g_bond = jraph.GraphsTuple(
        senders=senders,
        receivers=receivers,
        edges=edges_dict,
        nodes=nodes_dict,
        n_edge=n_edge,
        n_node=n_node,
        globals=globals
    )

    return g_bond

    
def load_data_from_parquet(graph_path, prior_graph_path, load_prior_graphs=True, debug=False):
    graphs = pl.read_parquet(graph_path)
    if debug: graphs = graphs.head(1000)

    graphs = [
        get_graph_from_row(row)
        for row in tqdm(graphs.iter_rows(named=True), total=graphs.shape[0])
    ]

    if not load_prior_graphs:
        return graphs

    prior_graphs = pl.read_parquet(prior_graph_path)
    if debug: prior_graphs = prior_graphs.head(1000)

    prior_graphs = [
        get_prior_graph_from_row(row)
        for row in tqdm(prior_graphs.iter_rows(named=True), total=prior_graphs.shape[0])
    ]

    cond_graphs = [
        cutoff_graph_to_bond_graph_old(g) 
        for g in tqdm(graphs, total=len(graphs))
    ]

    return graphs, prior_graphs, cond_graphs


def make_generator(graph_path, is_prior, debug=False):
    graphs = pl.scan_parquet(graph_path)
    if debug: graphs = graphs.limit(1000)
    for row in graphs.collect(streaming=True).iter_rows(named=True):
        if is_prior:
            yield get_prior_graph_from_row(row)
        else:
            yield get_graph_from_row(row)


def lazy_load_from_parquet(graph_path, prior_graph_path, load_prior_graphs=True, debug=False):
    gen_graph = make_generator(graph_path, is_prior=False, debug=debug)
    if not load_prior_graphs:
        return gen_graph
    else:
        gen_prior = make_generator(prior_graph_path, is_prior=True, debug=debug)
        return gen_graph, gen_prior

def compute_edges_tf(
    positions,
    cutoff: float
):
    num_atoms = tf.shape(positions)[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = tf.norm(displacements, axis=-1)
    mask = ~tf.eye(num_atoms, dtype=tf.bool)  # Get rid of self-connections.
    keep_edges = tf.where((distances < cutoff) & mask)
    centers = tf.cast(keep_edges[:, 0], dtype=tf.int32)  # center indices
    others = tf.cast(keep_edges[:, 1], dtype=tf.int32)  # neighbor indices
    return centers, others


def create_graph_latent(element, cutoff: float, split: str, store_extra: list = None) -> jraph.GraphsTuple:
    """Takes a data element and wraps relevant components in a GraphsTuple."""
    edges_dict = dict()
    globals_dict = dict()
    nodes_dict = dict()

    atomic_numbers = tf.cast(element['graph_atomic_numbers'], dtype=tf.int32)
    positions = tf.cast(element['graph_x1'], dtype=tf.float32)
    node_attr = tf.cast(element['graph_node_attr'], dtype=tf.float32)

    nodes_dict['atomic_numbers'] = atomic_numbers
    nodes_dict['node_attr'] = node_attr
    nodes_dict['x1'] = positions

    shortest_hops = tf.cast(element['graph_shortest_hops'], dtype=tf.int32)
    edges_dict['shortest_hops'] = shortest_hops

    if store_extra:
        for key in store_extra:
            globals_dict[key] = element[key]

    if split == 'test_small':
        smiles = tf.cast(element['smiles'], dtype=tf.string)
        rdkit_pickle = tf.cast(element['rdkit_pickle'], dtype=tf.string)
    
        globals_dict['smiles'] = tf.reshape(smiles, (1,))
        globals_dict['rdkit_pickle'] = tf.reshape(rdkit_pickle, (1,))

    centers, others = compute_edges_tf(
        positions=positions,
        cutoff=cutoff
    )

    num_nodes = tf.shape(node_attr)[0]
    num_edges = tf.shape(centers)[0]

    return jraph.GraphsTuple(
        n_node=tf.reshape(num_nodes, (1,)),
        n_edge=tf.reshape(num_edges, (1,)),
        receivers=centers,
        senders=others,
        nodes=nodes_dict,
        globals=globals_dict,
        edges=edges_dict,
    )


def create_graph_cond(element) -> jraph.GraphsTuple:

    """Takes a data element and wraps relevant components in a GraphsTuple."""
    edges_dict = dict()
    globals_dict = dict()
    nodes_dict = dict()

    atomic_numbers = tf.cast(element['graph_atomic_numbers'], dtype=tf.int32)
    node_attr = tf.cast(element['graph_node_attr'], dtype=tf.float32)

    nodes_dict['atomic_numbers'] = atomic_numbers
    nodes_dict['node_attr'] = node_attr

    edge_attr = tf.cast(element['cond_edge_attr'], dtype=tf.float32)

    edges_dict['edge_attr'] = edge_attr

    centers = tf.cast(element['cond_senders'], dtype=tf.int32)
    others = tf.cast(element['cond_receivers'], dtype=tf.int32)

    num_nodes = tf.shape(node_attr)[0]
    num_edges = tf.shape(centers)[0]

    return jraph.GraphsTuple(
        n_node=tf.reshape(num_nodes, (1,)),
        n_edge=tf.reshape(num_edges, (1,)),
        receivers=centers,
        senders=others,
        nodes=nodes_dict,
        globals=globals_dict,
        edges=edges_dict,
    )


def create_graph_prior(element) -> jraph.GraphsTuple:

    """Takes a data element and wraps relevant components in a GraphsTuple."""
    edges_dict = dict()
    globals_dict = dict()
    nodes_dict = dict()

    node_attr = tf.cast(element['prior_node_attr'], dtype=tf.float32)

    nodes_dict['node_attr'] = node_attr

    edge_attr = tf.cast(element['prior_edge_attr'], dtype=tf.float32)

    edges_dict['edge_attr'] = edge_attr

    # here we break the convention of senders and receivers
    # this graph is used during sampling from the harmonic prior
    senders = tf.cast(element['prior_senders'], dtype=tf.int32)
    receivers = tf.cast(element['prior_receivers'], dtype=tf.int32)

    num_nodes = tf.shape(node_attr)[0]
    num_edges = tf.shape(senders)[0]

    return jraph.GraphsTuple(
        n_node=tf.reshape(num_nodes, (1,)),
        n_edge=tf.reshape(num_edges, (1,)),
        receivers=receivers,
        senders=senders,
        nodes=nodes_dict,
        globals=globals_dict,
        edges=edges_dict,
    )


def create_graph_tuples(
        element,
        cutoff: float,
        split: str,
        to_numpy=False,
        store_extra: Optional[list] = None,
) -> Tuple[jraph.GraphsTuple, jraph.GraphsTuple, jraph.GraphsTuple]:
    
    graph_latent = create_graph_latent(element, cutoff=cutoff, split=split, store_extra=store_extra)
    graph_cond = create_graph_cond(element)
    graph_prior = create_graph_prior(element)

    def tonumpy(g):
        g = jraph.GraphsTuple(
            n_node=g.n_node.numpy(),
            n_edge=g.n_edge.numpy(),
            receivers=g.receivers.numpy(),
            senders=g.senders.numpy(),
            nodes={k: v.numpy() for k,v in g.nodes.items()},
            edges={k: v.numpy() for k,v in g.edges.items()},
            globals={k: v.numpy() for k,v in g.globals.items()},
        )
        return g
    
    if to_numpy:
        return (tonumpy(graph_latent), tonumpy(graph_cond), tonumpy(graph_prior))
    else:
        return (graph_latent, graph_cond, graph_prior)
