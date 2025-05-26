from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import jraph


def get_number_of_nodes(graph):
    return len(list(graph.nodes.values())[0])


def get_number_of_graphs(graph):
    """
    This function works for GraphsTuple and batched GraphsTuple.
    For the latter the padding graph(s) are also counted.
    """
    return len(graph.n_node)


def get_batch_segments(graph):
    num_graphs = get_number_of_graphs(graph)
    num_nodes = get_number_of_nodes(graph)

    batch_segments = jnp.repeat(
        jnp.arange(num_graphs), graph.n_node, total_repeat_length=num_nodes
    )

    return batch_segments


def is_batched_bool(graph):
    num_graphs = get_number_of_graphs(graph)
    if num_graphs <= 1:
        return False
    else:
        return True


def get_graph_padding_mask(graph):
    if is_batched_bool(graph) is True:
        return jraph.get_graph_padding_mask(graph)
    else:
        return jnp.array([True])


def get_node_padding_mask(graph):
    if is_batched_bool(graph) is True:
        return jraph.get_node_padding_mask(graph)
    else:
        num_nodes = get_number_of_nodes(graph)
        return jnp.array([True] * num_nodes)


def get_edge_padding_mask(graph):
    if is_batched_bool(graph) is True:
        return jraph.get_edge_padding_mask(graph)
    else:
        return jnp.array([True] * len(graph.senders))


def duplicate_graph(graph):
    deep_copy = jax.tree_util.tree_map(
        lambda x: x, graph
    )

    return deep_copy


def update_graph_positions(graph: jraph.GraphsTuple, new_positions):
    new_graph = duplicate_graph(graph)

    new_graph.nodes['positions'] = new_positions

    return new_graph


def move_graph_positions(graph: jraph.GraphsTuple, delta_positions):
    new_graph = duplicate_graph(graph)

    positions = new_graph.nodes['positions']

    new_positions = positions + delta_positions

    return update_graph_positions(new_graph, new_positions=new_positions)


def make_dummy_graph(num_atoms=2):
    return jraph.GraphsTuple(
        nodes={
            'positions': np.ones((num_atoms, 3), dtype=np.float32),
            'x0': np.ones((num_atoms, 3), dtype=np.float32),
            'x1': np.ones((num_atoms, 3), dtype=np.float32),
            'atomic_numbers': np.ones((num_atoms,), dtype=np.int32)
        },
        senders=np.arange(num_atoms),
        receivers=np.arange(num_atoms)[::-1],
        edges=dict(),
        globals=dict(),
        n_edge=np.array([len(np.arange(num_atoms))]),
        n_node=np.array([num_atoms])
    )


@dataclass
class BatchStatistics:
    graph: jraph.GraphsTuple
    num_graphs: int
    batch_segments: jnp.ndarray
    node_mask: jnp.ndarray
    node_mask_expanded: jnp.ndarray
    graph_mask: jnp.ndarray


def compute_batch_statistics(graph):
    num_graphs = get_number_of_graphs(graph)
    batch_segments = get_batch_segments(graph)
    node_mask = get_node_padding_mask(graph)
    graph_mask = get_graph_padding_mask(graph)

    if 'x1' in graph.nodes:
        x1 = graph.nodes['x1']

        node_mask_expanded = jnp.expand_dims(
            node_mask, 
            [x1.ndim - 1 - o for o in range(0, x1.ndim - 1)]
        )
    else:
       node_mask_expanded = None

    return BatchStatistics(
        graph=graph,
        num_graphs=num_graphs,
        batch_segments=batch_segments,
        node_mask=node_mask,
        node_mask_expanded=node_mask_expanded,
        graph_mask=graph_mask
    )
