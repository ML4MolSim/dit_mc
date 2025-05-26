import e3x
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from typing import Dict
from typing import Generator, Iterator, Tuple
from jaxtyping import PRNGKeyArray

from ..jraph_utils import get_batch_segments
from ..jraph_utils import get_number_of_graphs


_NUMBER_FIELDS = ("n_node", "n_edge", "n_graph", "n_edge_cond", "n_edge_prior")


def sample_times(
        rng: PRNGKeyArray,
        graph: jraph.GraphsTuple,
        tmin: float,
        tmax: float
):
    num_graphs = get_number_of_graphs(graph)

    graph.globals['t'] = jax.random.uniform(
        rng,
        minval=tmin,
        maxval=tmax,
        shape=(num_graphs,)
    )

    return graph


def rotate_graph(graph: jraph.GraphsTuple, rotation: jnp.ndarray):
    rotation_props = ["velocities", "positions", "z", "x0", "x1"]
    result = jax.tree_map(lambda x: x, graph)

    for p in rotation_props:
        # Rotate properties, if present.
        pval = result.nodes.get(p)
        pval_present = pval is not None
        if pval_present:
            pval_rot = jnp.einsum('bij, bj -> bi', rotation, pval)
            result.nodes[p] = pval_rot

    return result


# TODO: write a test for this
def rotation_augmentation(
    rng: PRNGKeyArray,
    graph: jraph.GraphsTuple,
    return_rotation: bool = False
):
    graph = jax.tree.map(lambda x: x, graph)

    num_graphs = get_number_of_graphs(graph)
    batch_segments = get_batch_segments(graph)

    rot = e3x.so3.random_rotation(key=rng, num=num_graphs)  # (num_graphs, 3, 3) if num_graphs > 1 else (3, 3)
    
    if num_graphs == 1:
        rot = rot[None]  # (1, 3, 3)

    rot = rot[batch_segments]  # (num_nodes, 3, 3)
    graph = rotate_graph(graph, rot)

    if return_rotation:
        return graph, rot
    else:
        return graph
    

def remove_com(
        graph: jraph.GraphsTuple,
        target: str
):
    batch_segments = get_batch_segments(graph)  # (num_nodes)
    num_graphs = get_number_of_graphs(graph)
    x = graph.nodes[target]  # (num_nodes)
    x_mean_per_graph = jraph.segment_mean(x, batch_segments, num_graphs)

    graph.nodes[target] = x - x_mean_per_graph[batch_segments]

    return graph


def aggregate_node_error(
        node_error,
        batch_segments,
        graph_mask,
        graph_weight=1.,
        scale=1.
):

    num_graphs = graph_mask.sum().astype(node_error.dtype)  # ()

    # sum up the l2_losses for node properties along the non-leading dimension. For e.g. scalar node quantities
    # this does not have any effect, but e.g. for vectorial and tensorial node properties one averages over all
    # additional non-leading dimension. E.g. for forces this corresponds to taking mean over x, y, z component.
    node_mean_squared = node_error.reshape(len(node_error), -1).mean(axis=-1)  # (num_nodes)

    per_graph_mse = jraph.segment_mean(
        data=node_mean_squared,
        segment_ids=batch_segments,
        num_segments=len(graph_mask)
    )  # (num_graphs)

    # Set contributions from padding graphs to zero.
    per_graph_mse = jnp.where(
        graph_mask,
        per_graph_mse,
        jnp.asarray(0., dtype=per_graph_mse.dtype)
    )  # (num_graphs)

    # Calculate weighted loss per graph
    per_graph_mse = graph_weight * per_graph_mse

    # Calculate mean and scale. Prevent the case of division by zero if no data is present at all.
    mse = scale * jnp.sum(per_graph_mse) / jnp.maximum(num_graphs, 1.)  # ()

    return mse


def make_optimizer(
        name: str = 'adam',
        optimizer_args: Dict = dict(),
        learning_rate: float = 1e-3,
        learning_rate_schedule: str = 'constant_schedule',
        learning_rate_schedule_args: Dict = dict(),
        gradient_clipping: str = 'identity',
        gradient_clipping_args: Dict = dict(),
        return_schedule_bool: bool = False,
):
    """Make optax optimizer.

    Args:
        name (str): Name of the optimizer. Defaults to the Adam optimizer.
        optimizer_args (dict): Arguments passed to the optimizer.
        learning_rate (float): Learning rate.
        learning_rate_schedule (str): Learning rate schedule. Defaults to no schedule, meaning learning rate is
            held constant.
        learning_rate_schedule_args (dict): Arguments for the learning rate schedule.
        gradient_clipping (str): Gradient clipping to apply.
        gradient_clipping_args (dict): Arguments to the gradient clipping to apply.
        return_schedule_bool (bool): return the lr schedule.
    Returns:

    """
    opt = getattr(optax, name)
    lr_schedule = getattr(optax, learning_rate_schedule)

    lr_schedule = lr_schedule(learning_rate, **learning_rate_schedule_args)
    opt = opt(lr_schedule, **optimizer_args)

    clip_transform = getattr(optax, gradient_clipping)
    clip_transform = clip_transform(**gradient_clipping_args)

    chained_opt = optax.chain(
        clip_transform,
        optax.zero_nans(),
        opt
    )

    if return_schedule_bool is True:
        return chained_opt, lr_schedule
    else:
        return chained_opt


# TODO: Move this to config as well
def get_optimizer(num_steps_total=200_000):
    opt, lr_schedule = make_optimizer(
        name='adamw',
        learning_rate=1e-5,
        learning_rate_schedule='warmup_cosine_decay_schedule',
        learning_rate_schedule_args=dict(
            peak_value=3e-4,
            warmup_steps=num_steps_total * 0.01,
            decay_steps=num_steps_total * 0.99,
        ),
        return_schedule_bool=True,
    )

    return opt, lr_schedule


def make_update_fn(
        optimizer,
        ema_bool: bool = True,
        ema_weight: float = 0.999
):
    if ema_bool is True:
        if ema_weight is None:
            raise ValueError(
                f"Exponential moving average requires a weight. Received {ema_bool=} and {ema_weight=}."
            )
    if ema_weight < 0:
        raise ValueError(
            f"Weight for exponential moving average must be larger than zero. Received {ema_weight=}."
        )

    def update_fn(params, grads, optimizer_state):
        updates, new_optimizer_state = optimizer.update(
            updates=grads,
            params=params,
            state=optimizer_state
        )

        new_params = optax.apply_updates(
            updates=updates,
            params=params
        )

        if ema_bool is True:
            new_params = optax.incremental_update(
                old_tensors=params,
                new_tensors=new_params,
                step_size=ema_weight
            )

        return new_params, new_optimizer_state

    return update_fn


def kabsch_algorithm(P, Q, return_rotation: bool = False):
    """
    Perform the Kabsch algorithm to find the optimal rotation matrix.

    Parameters:
        P (numpy.ndarray): A 2D array of shape (N, 3) representing the current coordinates.
        Q (numpy.ndarray): A 2D array of shape (N, 3) representing the reference coordinates.

    Returns:
        R (numpy.ndarray): The optimal rotation matrix of shape (3, 3).
    """
    # Step 1: Compute the covariance matrix
    H = np.dot(P.T, Q)

    # Step 2: Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Step 3: Compute the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Step 4: Ensure a proper rotation (determinant = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    if return_rotation is True:
        return P@R.T, R
    else:
        return P@R.T


def kabsch_align(graph, P, Q):
    num_graphs = get_number_of_graphs(graph)
    batch_segments = get_batch_segments(graph)

    # Step 0: Center the data
    P_mean_per_graph = jraph.segment_mean(
        P, segment_ids=batch_segments, num_segments=num_graphs
    ) # (num_graphs, 3)
    P_centered = P - P_mean_per_graph[batch_segments]  # (num_nodes, 3)

    Q_mean_per_graph = jraph.segment_mean(
        Q, segment_ids=batch_segments, num_segments=num_graphs
    ) # (num_graphs, 3)
    Q_centered = Q - Q_mean_per_graph[batch_segments]  # (num_nodes, 3)

    # Step 1: Compute the covariance matrix
    H_per_node = jnp.einsum(
        "bi,bj->bij", P_centered, Q_centered
    ) # (num_nodes, 3, 3)
    H = jraph.segment_sum(
        H_per_node, segment_ids=batch_segments, num_segments=num_graphs
    ) # (num_graphs, 3, 3)

    # Step 2: Perform Singular Value Decomposition (SVD)
    U, S, Vt = jnp.linalg.svd(H)  # (num_graphs, 3, 3)

    # Step 3: Ensure a proper rotation (determinant = +1)
    d = jnp.linalg.det(jnp.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1)))
    flip_mask = (d < 0.0).reshape(-1, 1, 1)
    flip_matrix = jnp.array([[1., 0., 0.], [0., 1., 0], [0., 0., -1.]])
    Vt = jnp.where(flip_mask, jnp.matmul(flip_matrix, Vt), Vt)

    # Step 4: Compute the optimal rotation matrices and translation vectors
    R = jnp.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1)) # (num_graphs, 3, 3)
    t = Q_mean_per_graph - jnp.einsum('ijk,ik->ij', R, P_mean_per_graph) # (num_graphs, 3)

    # Step 5: Align P based on the optimal R and t
    P = jnp.einsum("bi,bji->bj", P, R[batch_segments]) # (num_nodes, 3)
    P += t[batch_segments] # (num_nodes, 3)
    return P


def _get_graph_size(graph, graph_cond, graph_prior):
    n_node = np.sum(graph.n_node)
    n_edge = len(graph.senders)
    n_graph = len(graph.n_node)
    n_edge_cond = len(graph_cond.senders)
    n_edge_prior = len(graph_prior.senders)
    return n_node, n_edge, n_graph, n_edge_cond, n_edge_prior


def _is_over_batch_size(graph, graph_cond, graph_prior, graph_batch_size):
    graph_size = _get_graph_size(graph, graph_cond, graph_prior)
    return any([x > y for x, y in zip(graph_size, graph_batch_size)])


def dynamically_batch_extended(
    graphs_tuple_iterator: Iterator[Tuple[jraph.GraphsTuple, jraph.GraphsTuple, jraph.GraphsTuple]],
    n_node: int,
    n_edge: int,
    n_graph: int,
    n_edge_cond: int,
    n_edge_prior: int,
  ) -> Generator[jraph.GraphsTuple, jraph.GraphsTuple, jraph.GraphsTuple]:
  """Dynamically batches trees with `jraph.GraphsTuples` up to specified sizes.

  Differences from `jraph.utils.dynamically_batch`:
  - This function returns 
    Tuple[jraph.GraphsTuple, jraph.GraphsTuple, jraph.GraphsTuple], where:
    * The first jraph.GraphsTuple holds a cutoff graph.
    * The second jraph.GraphsTuple holds a bond graph.
    * The third jraph.GraphsTuple holds a graph for sampling from harmonic prior.

  Elements of the `graphs_tuple_iterator` will be incrementally added to a batch
  until the limits defined by `n_node`, `n_edge` and `n_graph` are reached. This
  means each element yielded by this generator may have a differing number of
  graphs in its batch.

  Args:
    graphs_tuple_iterator: An iterator of 
      `Tuple[jraph.GraphsTuple, aph.GraphsTuple, jraph.GraphsTuple]`.
    n_node: The maximum number of nodes in a batch, at least the maximum sized
      graph + 1.
    n_edge: The maximum number of edges in a batch, at least the maximum sized
      graph.
    n_graph: The maximum number of graphs in a batch, at least 2.

  Yields:
    A `Tuple[jraph.GraphsTuple, jraph.GraphsTuple, jraph.GraphsTuple]` batch of graphs.

  Raises:
    ValueError: if the number of graphs is < 2.
    RuntimeError: if the `graphs_tuple_iterator` contains elements which are not
      `jraph.GraphsTuple`s.
    RuntimeError: if a graph is found which is larger than the batch size.
  """
  if n_graph < 2:
    raise ValueError('The number of graphs in a batch size must be greater or '
                     f'equal to `2` for padding with graphs, got {n_graph}.')
  valid_batch_size = (n_node - 1, n_edge, n_graph - 1, n_edge_cond, n_edge_prior)
  accumulated_graphs = []
  accumulated_graphs_cond = []
  accumulated_graphs_prior = []
  num_accumulated_nodes = 0
  num_accumulated_edges = 0
  num_accumulated_graphs = 0
  num_accumulated_edges_cond = 0
  num_accumulated_edges_prior = 0
  for (element, element_cond, element_prior) in graphs_tuple_iterator:

    if _is_over_batch_size(element, element_cond, element_prior, valid_batch_size):
      # First yield the batched graph so far if exists.
      if accumulated_graphs:
        batched_graph = jraph.batch_np(accumulated_graphs)
        batched_graph = jraph.pad_with_graphs(
            batched_graph, n_node, n_edge, n_graph)
        batched_graph_cond = jraph.batch_np(accumulated_graphs_cond)
        batched_graph_cond = jraph.pad_with_graphs(
            batched_graph_cond, n_node, n_edge_cond, n_graph)
        batched_graph_prior = jraph.batch_np(accumulated_graphs_prior)
        batched_graph_prior = jraph.pad_with_graphs(
            batched_graph_prior, n_node, n_edge_prior, n_graph)
        yield (batched_graph, batched_graph_cond, batched_graph_prior)

      # Then report the error.
      graph_size = _get_graph_size(element, element_cond, element_prior)
      graph_size = {k: v for k, v in zip(_NUMBER_FIELDS, graph_size)}
      batch_size = {k: v for k, v in zip(_NUMBER_FIELDS, valid_batch_size)}
      raise RuntimeError('Found graph bigger than batch size. Valid Batch '
                         f'Size: {batch_size}, Graph Size: {graph_size}')

    # If this is the first element of the batch, set it and continue.
    # Otherwise check if there is space for the graph in the batch:
    #   if there is, add it to the batch
    #   if there isn't, return the old batch and start a new batch.
    ( 
        element_nodes, 
        element_edges, 
        element_graphs, 
        element_edges_cond, 
        element_edges_prior
    ) = _get_graph_size(element, element_cond, element_prior)
    if not accumulated_graphs:
      accumulated_graphs = [element]
      accumulated_graphs_cond = [element_cond]
      accumulated_graphs_prior = [element_prior]
      num_accumulated_nodes = element_nodes
      num_accumulated_edges = element_edges
      num_accumulated_graphs = element_graphs
      num_accumulated_edges_cond = element_edges_cond
      num_accumulated_edges_prior = element_edges_prior
      continue
    else:
      if ((num_accumulated_graphs + element_graphs > n_graph - 1) or
          (num_accumulated_nodes + element_nodes > n_node - 1) or
          (num_accumulated_edges + element_edges > n_edge) or
          (num_accumulated_edges_cond + element_edges_cond > n_edge_cond) or
          (num_accumulated_edges_prior + element_edges_prior > n_edge_prior)):
        batched_graph = jraph.batch_np(accumulated_graphs)
        batched_graph = jraph.pad_with_graphs(
            batched_graph, n_node, n_edge, n_graph)
        batched_graph_cond = jraph.batch_np(accumulated_graphs_cond)
        batched_graph_cond = jraph.pad_with_graphs(
            batched_graph_cond, n_node, n_edge_cond, n_graph)
        batched_graph_prior = jraph.batch_np(accumulated_graphs_prior)
        batched_graph_prior = jraph.pad_with_graphs(
            batched_graph_prior, n_node, n_edge_prior, n_graph)
        yield (batched_graph, batched_graph_cond, batched_graph_prior)
        accumulated_graphs = [element]
        accumulated_graphs_cond = [element_cond]
        accumulated_graphs_prior = [element_prior]
        num_accumulated_nodes = element_nodes
        num_accumulated_edges = element_edges
        num_accumulated_graphs = element_graphs
        num_accumulated_edges_cond = element_edges_cond
        num_accumulated_edges_prior = element_edges_prior
      else:
        accumulated_graphs.append(element)
        accumulated_graphs_cond.append(element_cond)
        accumulated_graphs_prior.append(element_prior)
        num_accumulated_nodes += element_nodes
        num_accumulated_edges += element_edges
        num_accumulated_graphs += element_graphs
        num_accumulated_edges_cond += element_edges_cond
        num_accumulated_edges_prior += element_edges_prior

  # We may still have data in batched graph.
  if accumulated_graphs:
    batched_graph = jraph.batch_np(accumulated_graphs)
    batched_graph = jraph.pad_with_graphs(
      batched_graph, n_node, n_edge, n_graph)
    batched_graph_cond = jraph.batch_np(accumulated_graphs_cond)
    batched_graph_cond = jraph.pad_with_graphs(
            batched_graph_cond, n_node, n_edge_cond, n_graph)
    batched_graph_prior = jraph.batch_np(accumulated_graphs_prior)
    batched_graph_prior = jraph.pad_with_graphs(
        batched_graph_prior, n_node, n_edge_prior, n_graph)
    yield (batched_graph, batched_graph_cond, batched_graph_prior)
