import e3x
import flax.linen as nn
import functools
import jax.numpy as jnp

from typing import Optional
from typing import Dict

from .base import BaseTimeEmbedding
from .base import BaseNodeEmbedding
from .base import BaseEdgeEmbedding
from .utils import GaussianRandomFourierFeatures
from .utils import MLP
from .utils import promote_to_e3x
from .utils import get_activation_fn
from .utils import get_pos_indices
from .utils import get_index_embedding
from .utils import broadcast_equivariant_multiplication


class RandomFourierFeaturesTimeEmbedding(BaseTimeEmbedding):
    num_features: int

    @nn.compact
    def __call__(self, time_latent, **kwargs):

        if time_latent.ndim > 1:
            raise ValueError(
                f'latent times must be an array of single dimension. '
                f'received shape {time_latent.shape}.'
            )

        features_time = GaussianRandomFourierFeatures(
            features=self.num_features
        )(
            jnp.expand_dims(time_latent, axis=-1)
        )  # (num_nodes, num_features)

        features_time = promote_to_e3x(features_time)  # (num_nodes, 1, 1, num_features)

        return features_time


class TimeEmbedding(BaseTimeEmbedding):
    num_features: int
    num_features_fourier: int = None
    activation_fn: str = 'silu'

    @nn.compact
    def __call__(self, time_latent, *args, **kwargs):

        num_features_fourier = self.num_features // 2 if self.num_features_fourier is None else self.num_features_fourier

        ff = GaussianRandomFourierFeatures(
            features=num_features_fourier
        )(
            jnp.expand_dims(time_latent, axis=-1)
        )  # (num_nodes, num_features)

        features_time = MLP(
            num_layers=2,
            activation_fn=self.activation_fn,
            num_features=self.num_features,
            use_bias=True
        )(
            ff
        )  # (num_nodes, 1, 1, num_features)

        features_time = promote_to_e3x(features_time)  # (num_nodes, 1, 1, num_features)

        return features_time


class AtomicTypeNodeEmbedding(BaseNodeEmbedding):
    num_features: int

    @nn.compact
    def __call__(self, graph, **kwargs):

        atomic_numbers = graph.nodes['atomic_numbers']

        features_node = e3x.nn.Embed(
            features=self.num_features,
            num_embeddings=119
        )(
            atomic_numbers
        )  # (num_nodes, 1, 1, num_features)

        return features_node


class NodeAttributeEmbedding(BaseNodeEmbedding):
    num_features: int
    activation_fn: str

    @nn.compact
    def __call__(self, graph, **kwargs):

        node_attributes = graph.nodes['node_attr']  # (num_atoms, node_attr_dim)
        features_nodes = MLP(
            num_layers=2,
            num_features=self.num_features,
            activation_fn=self.activation_fn
        )(
           node_attributes
        )  # (num_nodes, num_features)

        features_nodes = nn.LayerNorm()(
            features_nodes
        )  # (num_nodes, num_features)

        return promote_to_e3x(features_nodes)  # (num_nodes, 1, 1, num_features)


class EdgeAttributeEmbedding(BaseEdgeEmbedding):
    num_features: int
    activation_fn: str

    @nn.compact
    def __call__(self, graph, **kwargs):

        edge_attributes = graph.edges['edge_attr']  # (num_edges, edge_attr_dim)

        features_edges = MLP(
            num_layers=2,
            num_features=self.num_features,
            activation_fn=self.activation_fn
        )(
           edge_attributes
        )  # (num_edges, num_features)

        features_edges = nn.LayerNorm()(
            features_edges
        )  # (num_edges, num_features)

        return promote_to_e3x(features_edges)  # (num_edges, 1, 1, num_features)


class DiTNodeEmbed(BaseNodeEmbedding):
    num_features: int
    activation_fn: str
    self_conditioning_bool: bool
    positional_encoding_bool: bool
    positional_embedding_bool: bool

    @nn.compact
    def __call__(self, graph, **kwargs):
        atomic_numbers = graph.nodes['atomic_numbers']

        h = nn.Embed(
            features=self.num_features,
            num_embeddings=119
        )(
            atomic_numbers
        )  # (num_nodes, num_features)

        if self.self_conditioning_bool is True:
            self_cond = graph.nodes['self_cond']  # (num_nodes, 3) or (num_nodes, 6)

            sc = MLP(
                num_features=self.num_features,
                num_layers=2,
                use_bias=False,
                activation_fn=self.activation_fn
            )(
                self_cond
            )  # (num_nodes, num_features)

            h += sc  # (num_nodes, num_features)

        if self.positional_encoding_bool is True:
            n_node = graph.n_node
            num_nodes = len(graph.nodes['positions'])
            indices = get_pos_indices(n_node, num_nodes)  # (num_nodes,)
            h += get_index_embedding(indices, self.num_features)  # (num_nodes, num_features)

        if self.positional_embedding_bool is True:
            positions = graph.nodes['positions']  # (num_nodes, 3)

            p = MLP(
                num_features=self.num_features,
                num_layers=2,
                use_bias=False,
                activation_fn=self.activation_fn
            )(
                positions
            )  # (num_nodes, num_features)

            h += p  # (num_nodes, num_features)

        return promote_to_e3x(h)  # (num_nodes, 1, 1, num_features)


class DiTEdgeEmbed(BaseNodeEmbedding):
    num_features: int
    activation_fn: str

    embed_distances_bool: bool = True
    embed_shortest_hops_bool: bool = True

    radial_basis_bool: Optional[bool] = None
    num_radial_basis: Optional[int] = None
    max_frequency: Optional[int] = None

    def setup(self):
        if self.radial_basis_bool is True:
            
            if self.num_radial_basis is None:
                raise ValueError(
                    f'`num_radial_basis` must be provided if `radial_basis_bool` is True.'
                )
            
            if self.max_frequency is None:
                raise ValueError(
                    f'`max_frequency` must be provided if `radial_basis_bool` is True.'
                )
            
            if self.num_radial_basis <= 1:
                raise ValueError(
                    f'`num_radial_basis` must be greater than one. '
                    f'received {self.num_radial_basis}.'
                )

    @nn.compact
    def __call__(self, graph, **kwargs):

        num_edges = len(graph.senders)

        if self.embed_distances_bool == True:
            positions = graph.nodes['positions']  # (num_nodes, 3)

            src_idx = graph.senders  # (num_edges,)
            dst_idx = graph.receivers  # (num_edges,)

            # Calculate the displacements.
            displacements = positions[src_idx] - positions[dst_idx]  # (num_pairs, 3)

            if self.radial_basis_bool is True:
                displacements = e3x.nn.basic_fourier(
                    jnp.expand_dims(displacements, axis=-1),
                    num=self.num_radial_basis,
                    limit=(self.num_radial_basis - 1) * jnp.pi / self.max_frequency
                ).reshape(len(displacements), -1)  # (num_pairs, 3*num_radial_basis)
        
            re = MLP(
                num_features=self.num_features,
                num_layers=2,
                use_bias=False,
                activation_fn=self.activation_fn
            )(
                displacements
            )  # (num_pairs, num_features)
        else:
            re = jnp.zeros(
                (num_edges, self.num_features),
                dtype=graph.nodes['positions'].dtype
            )
        
        if self.embed_shortest_hops_bool:
            re_shortest_hops = nn.Embed(
                num_embeddings=512, 
                features=self.num_features,
            )(
                graph.edges['shortest_hops']
            ) # (num_pairs, num_features)

            re_shortest_hops = MLP(
                num_features=self.num_features,
                num_layers=2,
                use_bias=True,  # we need a bias here s.t. the output is non-zero in case of CFG
                activation_fn=self.activation_fn
            )(
                re_shortest_hops
            )  # (num_pairs, num_features)

            cond_scaling = graph.edges['cond_scaling'] # (num_edges)
            cond_scaling = cond_scaling[:, jnp.newaxis] # (num_edges, 1)

            re += re_shortest_hops * cond_scaling

        return promote_to_e3x(re)  # (num_edges, 1, 1, num_features)


class RadialSphericalEdgeEmbedding(BaseEdgeEmbedding):
    cutoff: float
    max_degree: int
    activation_fn: str
    embed_shortest_hops_bool: bool = False
    scale_spherical_basis_with_shortest_hops_bool: bool = False

    radial_basis: str = 'reciprocal_bernstein'
    num_radial_basis: int = 32
    radial_basis_kwargs: Optional[Dict] = None

    cutoff_fn: str = 'smooth_cutoff'

    @nn.compact
    def __call__(self, graph, **kwargs):
        src_idx = graph.senders
        dst_idx = graph.receivers

        num_edges = len(src_idx)

        positions = graph.nodes['positions']

        # Calculate the displacements.
        displacements = positions[src_idx] - positions[dst_idx]  # (num_pairs, 3)

        basis = e3x.nn.basis(
            displacements,
            num=self.num_radial_basis,
            max_degree=self.max_degree,
            radial_fn=functools.partial(
                getattr(e3x.nn, self.radial_basis),
                **self.radial_basis_kwargs if self.radial_basis_kwargs is not None else {}
            ),
            cutoff_fn=functools.partial(
                getattr(e3x.nn, self.cutoff_fn),
                cutoff=self.cutoff
            ) if self.cutoff_fn is not None else None,
        )  # (num_pairs, 1, (max_degree+1)**2, num_basis_functions)

        if self.embed_shortest_hops_bool:

            if self.scale_spherical_basis_with_shortest_hops_bool:
                num_features = self.num_radial_basis * (self.max_degree + 1)
            else:
                num_features = self.num_radial_basis

            re_shortest_hops = nn.Embed(
                num_embeddings=512, 
                features=num_features,
            )(
                graph.edges['shortest_hops']
            )  # (num_pairs, num_features)

            # we want to learn a scaling or bias so we init the output with zero
            re_shortest_hops = MLP(
                num_features=num_features,
                num_layers=2,
                use_bias=False,
                activation_fn=self.activation_fn,
                output_is_zero_at_init=True,
            )(
                re_shortest_hops
            )  # (num_pairs, num_features)

            cond_scaling = graph.edges['cond_scaling'] # (num_edges)
            cond_scaling = cond_scaling[:, jnp.newaxis] # (num_edges, 1)

            re_shortest_hops = cond_scaling * re_shortest_hops

            re_shortest_hops = promote_to_e3x(
                re_shortest_hops
            )  # (num_pairs, 1, 1, num_features)

            if self.scale_spherical_basis_with_shortest_hops_bool:
                re_shortest_hops = re_shortest_hops.reshape(
                    num_edges, 1, (self.max_degree + 1), self.num_radial_basis
                )  # (num_pairs, 1, max_degree+1, num_features)

                basis = broadcast_equivariant_multiplication(
                    factor=1 + re_shortest_hops,
                    tensor=basis
                )  # (num_pairs, 1, (max_degree+1)**2, num_features)
            else:
                # hack to bring to correct shape
                re_shortest_hops = e3x.nn.add(
                    re_shortest_hops, 
                    jnp.zeros((*basis.shape[:3], re_shortest_hops.shape[-1]))
                )  # (num_pairs, 1, (max_degree+1)**2, num_features)

                # does the same as this code:
                # x = re_shortest_hops
                # y = basis
                # _x = jnp.zeros_like(y)
                # x = _x.at[:, 0, 0, :].set(jnp.squeeze(x, axis=(1, 2)))

                basis = e3x.nn.add(
                    basis, re_shortest_hops
                )  # (num_pairs, 1, (max_degree+1)**2, num_features)

        return basis
