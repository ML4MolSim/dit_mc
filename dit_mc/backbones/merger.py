import flax.linen as nn
import jax.numpy as jnp

from .base import BaseMerger
from .base import FeatureRepresentations
from .utils import E3MLP
from .utils import equivariant_concatenation
from .utils import EquivariantLayerNorm

from jaxtyping import Array


class IdentityMerger(BaseMerger):
    activation_fn: str = None

    @nn.compact
    def __call__(
            self,
            features: FeatureRepresentations,
            **kwargs
    ):

        return features


class MLPMerger(BaseMerger):
    activation_fn: str
    apply_to: str = 'nodes'  # 'edges' 'nodes_and_edges'

    @nn.compact
    def __call__(
            self,
            features: FeatureRepresentations,
            **kwargs
    ):
        if self.apply_to == 'nodes':
            y = features.nodes

            y = E3MLP(
                activation_fn=self.activation_fn
            )(
                y
            )

            y = EquivariantLayerNorm()(y)

            return FeatureRepresentations(
                nodes=y, edges=features.edges
            )

        elif self.apply_to == 'edges':
            y = features.edges

            y = E3MLP(
                activation_fn=self.activation_fn
            )(
                y
            )

            y = EquivariantLayerNorm()(y)

            return FeatureRepresentations(
                nodes=features.nodes, edges=y
            )

        elif self.apply_to == 'nodes_and_edges':

            y = features.nodes
            z = features.edges

            y = E3MLP(
                activation_fn=self.activation_fn
            )(
                y
            )

            y = EquivariantLayerNorm()(y)

            z = E3MLP(
                activation_fn=self.activation_fn
            )(
                z
            )

            z = EquivariantLayerNorm()(z)

            return FeatureRepresentations(
                    nodes=y, edges=z
                )

        else:
            raise ValueError(
                f'{self.apply_to=} is not a valid option for merger targets. '
                f'Try one of `nodes`, `edges`, `nodes_and_edges`'
            )


class TimeMerger(BaseMerger):
    activation_fn: str
    apply_to: str = 'nodes'

    @nn.compact
    def __call__(
            self,
            features: FeatureRepresentations,
            features_time: Array,
            **kwargs
    ):
        if self.apply_to != 'nodes':
            raise NotImplementedError(
                f"time merger at the moment only supported for apply_to = 'nodes'. "
                f"received {self.apply_to=}"
            )

        features_node = features.nodes

        assert features_time is not None, f'features_time must not be None. Received {features_time=}'
        assert features_node.shape[0] == features_time.shape[0]
        assert features_node.shape[-1] == features_time.shape[-1]

        num_features = features_node.shape[-1]

        y = equivariant_concatenation(
            features_node,
            features_time
        )  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

        y = E3MLP(
            activation_fn=self.activation_fn,
            num_features=num_features,
            num_layers=2,
            use_bias=True
        )(
            y
        )  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

        y = EquivariantLayerNorm()(y)  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

        return FeatureRepresentations(
            nodes=y,
            edges=features.edges
        )


class CondMerger(BaseMerger):
    activation_fn: str
    apply_to: str = 'nodes'

    @nn.compact
    def __call__(
            self,
            features: FeatureRepresentations,
            features_cond: FeatureRepresentations,
            **kwargs
    ):

        assert features_cond is not None, f'features_cond must not be None. Received {features_cond=}'

        if self.apply_to == 'nodes':
            features_node = features.nodes
            features_cond_node = features_cond.nodes

            num_features = features_node.shape[-1]

            y = equivariant_concatenation(
                    features_node,
                    features_cond_node
            )  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

            y = E3MLP(
                activation_fn=self.activation_fn,
                num_features=num_features,
                num_layers=2,
                use_bias=True
            )(
                y
            )  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

            y = EquivariantLayerNorm()(y)  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

            return FeatureRepresentations(
                nodes=y,
                edges=features.edges
            )

        elif self.apply_to == 'edges':
            features_edges = features.edges
            features_cond_edges = features_cond.edges

            num_features = features_edges.shape[-1]

            y = equivariant_concatenation(
                features_edges,
                features_cond_edges
            )  # (num_edges, P, (max_degree+1)**2, 2*num_features)

            y = E3MLP(
                activation_fn=self.activation_fn,
                num_features=num_features,
                num_layers=2,
                use_bias=True
            )(
                y
            )  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

            y = EquivariantLayerNorm()(y)  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

            return FeatureRepresentations(
                nodes=features.nodes,
                edges=y
            )
        elif self.apply_to == 'nodes_and_edges':
            features_node = features.nodes
            features_cond_node = features_cond.nodes

            num_features_node = features_node.shape[-1]

            y = equivariant_concatenation(
                    features_node,
                    features_cond_node
            )  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

            y = E3MLP(
                activation_fn=self.activation_fn,
                num_features=num_features_node,
                num_layers=2,
                use_bias=True
            )(
                y
            )  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

            y = EquivariantLayerNorm()(y)  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

            features_edges = features.edges
            features_cond_edges = features_cond.edges

            num_features_edges = features_edges.shape[-1]

            z = equivariant_concatenation(
                features_edges,
                features_cond_edges
            )  # (num_edges, P, (max_degree+1)**2, 2*num_features)

            z = E3MLP(
                activation_fn=self.activation_fn,
                num_features=num_features_edges,
                num_layers=2,
                use_bias=True
            )(
                z
            )  # (num_edges, P, (max_degree+1)**2, 2*num_features)

            z = EquivariantLayerNorm()(z)  # (num_edges, P, (max_degree+1)**2, 2*num_features)

            return FeatureRepresentations(
                nodes=y,
                edges=z
            )
        else:
            raise ValueError(
                f'{self.apply_to=} is not a valid option for merger targets. '
                f'Try one of `nodes`, `edges`, `nodes_and_edges`'
            )


class TimeCondMerger(BaseMerger):
    activation_fn: str
    apply_to: str = 'nodes'

    @nn.compact
    def __call__(
            self,
            features: FeatureRepresentations,
            features_cond: FeatureRepresentations,
            features_time: Array,
            **kwargs
    ):
        if self.apply_to != 'nodes':
            raise NotImplementedError(
                f"time merger at the moment only supported for apply_to = 'nodes'. "
                f"received {self.apply_to=}"
            )

        assert features_time is not None, f'features_time must not be None. Received {features_time=}'
        assert features_cond is not None, f'features_cond must not be None. Received {features_cond=}'

        features_node = features.nodes  # (num_nodes, P, (max_degree + 1)**2, num_features)
        features_cond_node = features_cond.nodes # (num_nodes, P_cond, (max_degree_cond + 1)**2, num_features_cond)

        assert features_node.shape[0] == features_time.shape[0]
        assert features_cond_node.shape[0] == features_time.shape[0]

        num_features = features_node.shape[-1]

        y1 = equivariant_concatenation(
            features_node,
            features_time
        )  # (num_atoms, P, (max_degree+1)**2, num_features_time + num_features)
        # max_degree and P will be max(max_degree_node, max_degree_time) and max(P_node, P_time)
        y = equivariant_concatenation(
            y1,
            features_cond_node
        )  # (num_atoms, P, (max_degree+1)**2, num_features_time + num_features + num_features_cond)

        y = E3MLP(
            activation_fn=self.activation_fn,
            num_features=num_features,
            num_layers=2,
            use_bias=True
        )(
            y
        )  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

        y = EquivariantLayerNorm()(y)  # (num_atoms, P, (max_degree+1)**2, 2*num_features)

        return FeatureRepresentations(
            nodes=y,
            edges=features.edges
        )


#
# class SimpleMerger(BaseMerger):
#     activation_fn: str
#     output: str
#
#     @nn.compact
#     def __call__(
#             self,
#             node_features_tau,
#             node_features_cond
#     ):
#
#         if node_features_tau.shape != node_features_cond.shape:
#             raise ValueError(
#                 f'Shapes of latent and conditioning features must be equal. '
#                 f'Received {node_features_tau.shape=} and {node_features_cond.shape=}.'
#             )
#
#         num_features = node_features_tau.shape[-1]
#
#         y = node_features_cond + node_features_tau
#
#         z = MLP(
#             num_layers=2,
#             activation=self.activation_fn,
#             use_bias=True,
#             features=num_features
#         )(
#             y
#         )  # (num_nodes, features)
#
#         y = y + z
#
#         if self.output == 'drift' or self.output == 'noise':
#
#             drift_or_noise = MLP(
#                 num_layers=2,
#                 activation=self.activation_fn,
#                 use_bias=True,
#                 features=num_features
#             )(
#                 y
#             )  # (num_nodes, features)
#             drift_or_noise = nn.Dense(features=3, use_bias=False)(drift_or_noise)
#
#             return drift_or_noise  # (num_nodes, 3)
#
#         elif self.output == 'drift_and_noise':
#             drift = MLP(
#                 num_layers=2,
#                 activation=self.activation_fn,
#                 use_bias=True,
#                 features=num_features
#             )(
#                 y
#             )  # (num_nodes, features)
#
#             drift = nn.Dense(features=3, use_bias=False)(drift)
#
#             noise = MLP(
#                 num_layers=2,
#                 activation=self.activation_fn,
#                 use_bias=True,
#                 features=num_features
#             )(
#                 y
#             )  # (num_nodes, features)
#             noise = nn.Dense(features=3, use_bias=False)(noise)
#
#             return drift, noise  # (num_nodes, 3), (num_nodes, 3)
#
#         else:
#             raise ValueError(
#                 f'Unknown output option. Possible inputs are `drift`, `noise` or `drift_and_noise`. '
#                 f'Received {self.output=}.'
#             )
#
#
# class EquivariantMerger(BaseMerger):
#     activation_fn: str
#     output: str
#
#     @nn.compact
#     def __call__(
#             self,
#             node_features_tau,
#             node_features_cond
#     ):
#
#         activation_fn = getattr(e3x.nn, self.activation_fn)
#
#         if node_features_tau.shape != node_features_cond.shape:
#             raise ValueError(
#                 f'Shapes of latent and conditioning features must be equal. '
#                 f'Received {node_features_tau.shape=} and {node_features_cond.shape=}.'
#             )
#
#         num_features = node_features_tau.shape[-1]
#
#         y = e3x.nn.add(
#             node_features_tau,
#             node_features_cond
#         )  # (num_atoms, 1 or 2, (max_degree+1)**2, features)
#
#         z = e3x.nn.add(
#             y,
#             e3x.nn.Dense(num_features)(activation_fn(e3x.nn.Dense(features=num_features)(y)))
#         )  # (num_atoms, 1 or 2, (max_degree+1)**2, features)
#
#         y = e3x.nn.add(y, z)
#
#         y = e3x.nn.TensorDense(
#             features=num_features,
#             include_pseudotensors=False,
#             max_degree=1
#         )(
#             y
#         )
#
#         if self.output == 'noise' or self.output == 'drift':
#             drift_or_noise = e3x.nn.Dense(num_features)(activation_fn(e3x.nn.Dense(features=num_features)(y)))
#
#             drift_or_noise = drift_or_noise[:, 0, 1:4, 0]  # (num_nodes, 3)
#
#             return drift_or_noise
#         else:
#             drift = e3x.nn.Dense(num_features)(activation_fn(e3x.nn.Dense(features=num_features)(y)))
#             drift = drift[:, 0, 1:4, 0]  # (num_nodes, 3)
#
#             noise = e3x.nn.Dense(num_features)(activation_fn(e3x.nn.Dense(features=num_features)(y)))
#             noise = noise[:, 0, 1:4, 0]  # (num_nodes, 3)
#
#             return drift, noise
