"""DiT models for the GEOM dataset."""

import jax.numpy as jnp

from typing import Dict

from ..backbones import embedding
from ..backbones import encoder
from ..backbones import merger
from ..backbones import neural_network_layer
from ..backbones import generative_model
from ..backbones import base
from ..backbones import readout


def make_graph_mesh_net_encoder(
        num_layers: int,
        num_features: int,
        activation_fn: str
):

    node_embedding = embedding.NodeAttributeEmbedding(
        num_features=num_features,
        activation_fn=activation_fn
    )

    edge_embedding = embedding.EdgeAttributeEmbedding(
        num_features=num_features,
        activation_fn=activation_fn
    )

    layers = []

    # The encoder is simply a stack of neural networks layers.
    for n in range(num_layers):
        layers.append(
            neural_network_layer.MeshGraphNetLayer(
                num_edge_features=num_features,
                num_node_features=num_features,
                activation_fn=activation_fn
            )
        )

    return encoder.EncoderModel(
        node_embedding=node_embedding,
        edge_embedding=edge_embedding,
        layers=layers
    )


def make_molecular_DiT(
        num_layers: int,
        num_heads: int,
        num_features_head: int,
        mgn_num_features: int,
        mgn_num_layers: int,
        mgn_activation_fn: str,
        num_features_mlp: int = None,
        activation_fn_mlp: str = 'gelu',
        activation_fn: str = 'silu',
        absolute_positional_embedding_bool: bool = True,
        relative_positional_embedding_bool: bool = False,
        rpe_radial_basis_bool: bool = False,
        rpe_num_radial_basis: int = 8,
        rpe_max_frequency: float = 2 * jnp.pi,
        self_conditioning_bool: bool = False,
        positional_encoding_bool: bool = False,
        embed_shortest_hops_bool: bool = False,
        act_dense_correct_bool: bool = False,
        output: str = 'drift_and_noise',
        name: str = "DiT",
):

    num_features = num_heads * num_features_head

    if num_features_mlp is None:
        num_features_mlp = 4 * num_features

    # Conditioning is not optional for GEOM dataset
    encoder_cond = make_graph_mesh_net_encoder(
        num_layers=mgn_num_layers,
        num_features=mgn_num_features,
        activation_fn=mgn_activation_fn
    )

    # Mergers are not required for DiT-like architectures.
    mergers = [merger.IdentityMerger for _ in range(num_layers)]

    # Time embedding
    time_embedding = embedding.TimeEmbedding(
        num_features=num_features,
        activation_fn=activation_fn
    )

    # Node embedding
    node_embedding = embedding.DiTNodeEmbed(
        num_features=num_features,
        activation_fn=activation_fn,
        self_conditioning_bool=self_conditioning_bool,
        positional_encoding_bool=positional_encoding_bool,
        positional_embedding_bool=absolute_positional_embedding_bool
    )

    # Relative embedding.
    relative_embedding_bool = relative_positional_embedding_bool or embed_shortest_hops_bool
    if relative_embedding_bool:
        edge_embedding = embedding.DiTEdgeEmbed(
            num_features=num_features,
            activation_fn=activation_fn,
            radial_basis_bool=rpe_radial_basis_bool,
            num_radial_basis=rpe_num_radial_basis,
            max_frequency=rpe_max_frequency,
            embed_distances_bool=relative_positional_embedding_bool,
            embed_shortest_hops_bool=embed_shortest_hops_bool,
        )
    else:
        edge_embedding = None

    readout_block = readout.SimpleReadout(
        activation_fn=activation_fn,
        output=output
    )

    layers = []

    for n in range(num_layers):
        layers.append(
            base.GenerativeLayer(
                merger=mergers[n](activation_fn=activation_fn),
                encoder=neural_network_layer.DiTLayer(
                    num_heads=num_heads,
                    num_features_mlp=num_features_mlp,
                    activation_fn_mlp=activation_fn_mlp,
                    activation_fn=activation_fn,
                    relative_embedding_qk_bool=relative_embedding_bool,
                    relative_embedding_v_bool=relative_embedding_bool,
                    act_dense_correct_bool=act_dense_correct_bool,
                )
            )
        )

    gm = generative_model.GenerativeModel(
        time_embedding=time_embedding,
        node_embedding=node_embedding,
        edge_embedding=edge_embedding,
        layers=layers,
        readout=readout_block,
        conditioner=encoder_cond,
        conditioning_bool=True,
        name=name
    )

    return gm


def make_molecular_DiT_SO3(
        num_layers: int,
        num_heads: int,
        num_features_head: int,
        cutoff: float,
        mgn_num_features: int,
        mgn_num_layers: int,
        mgn_activation_fn: str,
        max_degree: int = 1,
        num_features_mlp: int = None,
        activation_fn_mlp: str = 'gelu',
        activation_fn: str = 'silu',
        include_pseudotensors: bool = True,
        radial_basis: str = 'reciprocal_bernstein',
        num_radial_basis: int = 64,
        radial_basis_kwargs: Dict = None,
        cutoff_fn: str = 'cosine_cutoff',
        self_conditioning_bool: bool = False,
        positional_encoding_bool: bool = False,
        embed_shortest_hops_bool: bool = False,
        scale_spherical_basis_with_shortest_hops_bool: bool = False,
        act_dense_correct_bool: bool = False,
        output: str = 'drift_and_noise',
        name: str = "SO3_DiT",
):
    if cutoff is not None:
        raise NotImplementedError(
            f'Cutoff not None not supported yet. Received {cutoff=}'
        )

    if cutoff is None:
        if cutoff_fn is not None:
            raise ValueError(
                f'When no cutoff is used, cutoff_fn must also be None. '
                f'Received {cutoff=} and {cutoff_fn=}.'
            )

    num_features = num_heads * num_features_head

    if num_features_mlp is None:
        num_features_mlp = 4 * num_features

    # Conditioning is not optional for GEOM dataset
    encoder_cond = make_graph_mesh_net_encoder(
        num_layers=mgn_num_layers,
        num_features=mgn_num_features,
        activation_fn=mgn_activation_fn
    )

    # Mergers are not required for DiT-like architectures.
    mergers = [merger.IdentityMerger for _ in range(num_layers)]

    # Time embedding
    time_embedding = embedding.TimeEmbedding(
        num_features=num_features,
        activation_fn=activation_fn
    )

    # Invariant node embedding (uses only the atomic types)
    node_embedding = embedding.DiTNodeEmbed(
        num_features=num_features,
        activation_fn=activation_fn,
        self_conditioning_bool=self_conditioning_bool,
        positional_encoding_bool=positional_encoding_bool,
        positional_embedding_bool=False
    )

    # Relative, SO3 equivariant edge embedding.
    edge_embedding = embedding.RadialSphericalEdgeEmbedding(
        cutoff=cutoff,
        max_degree=max_degree,
        radial_basis=radial_basis,
        num_radial_basis=num_radial_basis,
        radial_basis_kwargs=radial_basis_kwargs,
        cutoff_fn=cutoff_fn,
        activation_fn=activation_fn, # only used in cases of `embed_shortest_hops_bool=True`
        embed_shortest_hops_bool=embed_shortest_hops_bool,
        scale_spherical_basis_with_shortest_hops_bool=scale_spherical_basis_with_shortest_hops_bool,
    )

    readout_block = readout.EquivariantReadout(
        activation_fn=activation_fn,
        output=output
    )

    layers = []

    for n in range(num_layers):
        layers.append(
            base.GenerativeLayer(
                merger=mergers[n](activation_fn=activation_fn),
                encoder=neural_network_layer.SO3DiTLayer(
                    num_heads=num_heads,
                    max_degree=max_degree,
                    include_pseudotensors=include_pseudotensors,
                    num_features_mlp=num_features_mlp,
                    activation_fn_mlp=activation_fn_mlp,
                    activation_fn=activation_fn,
                    act_dense_correct_bool=act_dense_correct_bool,
                )
            )
        )

    gm = generative_model.GenerativeModel(
        time_embedding=time_embedding,
        node_embedding=node_embedding,
        edge_embedding=edge_embedding,
        layers=layers,
        readout=readout_block,
        conditioner=encoder_cond,
        conditioning_bool=True,
        name=name
    )

    return gm
