import jax
import jax.numpy as jnp
import jraph
import numpy as np
from typing import Optional, Union, Tuple
from rdkit import Chem

from dit_mc.jraph_utils import get_number_of_nodes


BOND_TYPE_LIST = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"]


"""
Code adopted from
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/num_nodes.py
"""
def maybe_num_nodes(
    src_idx: jnp.ndarray,
    dst_idx: jnp.ndarray,
    num_nodes: Optional[int] = None,
) -> int:
    if num_nodes is not None:
        return num_nodes
    return max(
        int(src_idx.max()) + 1 if jnp.size(src_idx) > 0 else 0,
        int(dst_idx.max()) + 1 if jnp.size(dst_idx) > 0 else 0,
    )


"""
Code adopted from
https://github.com/pyg-team/pytorch_geometric/blob/main/torch_geometric/utils/loop.py
"""
def remove_self_loops(
    src_idx: jnp.ndarray,
    dst_idx: jnp.ndarray,
    edge_attr: Optional[jnp.ndarray] = None,
):
    mask = src_idx != dst_idx
    mask = jnp.nonzero(mask)[0]
    src_idx, dst_idx = src_idx[mask], dst_idx[mask]
    if edge_attr is None:
        return src_idx, dst_idx, None
    return src_idx, dst_idx, edge_attr[mask]


"""
Code adopted from
https://github.com/pyg-team/pytorch_geometric/blob/main/torch_geometric/utils/loop.py
"""
def compute_loop_attr(
    edge_attr: jnp.ndarray,
    num_nodes: int,
    fill_value: Optional[Union[float, jnp.ndarray]] = None,
) -> jnp.ndarray:

    if fill_value is None:
        size = (num_nodes, ) + edge_attr.shape[1:]
        return jnp.ones(size)

    elif isinstance(fill_value, (int, float)):
        size = (num_nodes, ) + edge_attr.shape[1:]
        return jnp.full(size, fill_value)

    elif isinstance(fill_value, jnp.ndarray):
        size = (num_nodes, ) + edge_attr.shape[1:]
        loop_attr = fill_value.astype(edge_attr.dtype)
        if edge_attr.ndim != loop_attr.ndim:
            loop_attr = loop_attr.squeeze()
        return jnp.broadcast_to(loop_attr, size)

    raise AttributeError("No valid 'fill_value' provided")


"""
Code adopted from
https://github.com/pyg-team/pytorch_geometric/blob/main/torch_geometric/utils/loop.py
"""
def add_self_loops(
    src_idx: jnp.ndarray,
    dst_idx: jnp.ndarray,
    edge_attr: Optional[jnp.ndarray] = None,
    fill_value: Optional[Union[float, jnp.ndarray]] = None,
    num_nodes: Optional[int] = None,
):
    N = maybe_num_nodes(src_idx, dst_idx, num_nodes)

    loop_idx = jnp.arange(0, N)
    full_src_idx = jnp.concatenate((src_idx, loop_idx), axis=0)
    full_dst_idx = jnp.concatenate((dst_idx, loop_idx), axis=0)

    if edge_attr is not None:
        loop_attr = compute_loop_attr(edge_attr, N, fill_value)
        edge_attr = jnp.concatentate([edge_attr, loop_attr], axis=0)

    return full_src_idx, full_dst_idx, edge_attr


"""
Code adopted from
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/laplacian.py
"""
def get_laplacian(
    src_idx: jnp.ndarray,
    dst_idx: jnp.ndarray,
    edge_weight: Optional[jnp.ndarray] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    src_idx, dst_idx, edge_weight = remove_self_loops(src_idx, dst_idx, edge_weight)

    if edge_weight is None:
        edge_weight = jnp.ones(src_idx.size, dtype=np.float32)

    num_nodes = maybe_num_nodes(src_idx, dst_idx, num_nodes)

    deg = jraph.segment_sum(edge_weight, src_idx, num_nodes)

    # L = D - A.
    src_idx, dst_idx, _ = add_self_loops(src_idx, dst_idx, num_nodes=num_nodes)
    edge_weight = jnp.concatenate((-edge_weight, deg), axis=0)
    return src_idx, dst_idx, edge_weight


"""
Code adopted from
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/functions.py
"""
def cumsum(x: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (1, 0)
    x = jnp.pad(x, pad_width, mode='constant', constant_values=0)
    return jnp.cumsum(x, axis=axis)


"""
Code adopted from
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/_to_dense_adj.py
"""
def to_dense_adj(
    src_idx: jnp.ndarray,
    dst_idx: jnp.ndarray,
    batch: Optional[jnp.ndarray] = None,
    edge_attr: Optional[jnp.ndarray] = None,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:

    if batch is None:
        max_index = max(
            int(src_idx.max()) + 1 if jnp.size(src_idx) > 0 else 0,
            int(dst_idx.max()) + 1 if jnp.size(dst_idx) > 0 else 0,
        )
        batch = jnp.zeros(max_index, dtype=np.int32)

    if batch_size is None:
        batch_size = int(batch.max()) + 1 if jnp.size(batch) > 0 else 0

    one = jnp.ones(batch.shape[0], dtype=np.int32)
    num_nodes = jraph.segment_sum(one, batch, batch_size)
    cum_nodes = cumsum(num_nodes)

    idx0 = batch[src_idx]
    idx1 = src_idx - cum_nodes[batch][src_idx]
    idx2 = dst_idx - cum_nodes[batch][dst_idx]

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    elif ((jnp.size(idx1) > 0 and idx1.max() >= max_num_nodes)
          or (jnp.size(idx2) > 0 and idx2.max() >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = jnp.ones(jnp.size(idx0))

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.shape)[1:]
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = jraph.segment_sum(edge_attr, idx, flattened_size)
    adj = adj.reshape(size)

    return adj


def get_mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol


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


def compute_eigh_laplacian(src_idx, dst_idx, num_nodes, alpha=1.0):

    edge_attr = jnp.ones_like(src_idx)
    edge_attr *= alpha

    src_idx, dst_idx, edge_weight = get_laplacian(
        src_idx,
        dst_idx,
        edge_attr,
        num_nodes=num_nodes,
    )

    H = to_dense_adj(
        src_idx,
        dst_idx,
        edge_attr=edge_weight,
        max_num_nodes=num_nodes,
    ).squeeze()

    D, P = jnp.linalg.eigh(H)

    return D, P


class GaussianPrior:
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, name="GaussianPrior"):
        self.name = name
        self.mu = mu
        self.sigma = sigma

    def sample(
        self,
        key: jax.random.PRNGKey,
        shape: Tuple[int, ...],
        **kwargs,
    ) -> jnp.ndarray:

        noise = jax.random.normal(key, shape=shape)
        sample = self.mu + self.sigma * noise

        return sample
    

class HarmonicPrior:
    def __init__(self, name="HarmonicPrior"):
        self.name = name

    def sample(
        self,
        key: jax.random.PRNGKey,
        shape: Tuple[int, ...],
        graph_prior: jraph.GraphsTuple,
    ) -> jnp.ndarray:

        num_nodes = get_number_of_nodes(graph_prior)

        senders = graph_prior.senders
        receivers = graph_prior.receivers
        node_attr = graph_prior.nodes['node_attr'] # (num_nodes)
        edge_attr = graph_prior.edges['edge_attr'] # (num_edges)

        noise = jax.random.normal(key, shape=shape) # (num_nodes, ...)
        scaled_noise = jnp.einsum('t, tn... -> tn...', node_attr, noise)
        scaled_noise = jnp.nan_to_num(scaled_noise)
        scaled_noise = jnp.take(scaled_noise, senders, axis=0) # (num_edges, ...)
        messages = jnp.einsum('t, tn... -> tn...', edge_attr, scaled_noise)
        sample = jraph.segment_sum(
            messages, receivers, num_segments=num_nodes) # (num_nodes, ...)

        return sample
