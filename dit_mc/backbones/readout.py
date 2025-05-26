import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from jaxtyping import Array

from dit_mc.jraph_utils import get_batch_segments, get_number_of_graphs, get_node_padding_mask

from .base import BaseReadout
from .base import FeatureRepresentations
from .utils import MLP, EquivariantLayerNorm
from .utils import get_max_degree_from_tensor_e3x
from .utils import get_activation_fn
from .utils import modulate_adaLN
from .utils import modulate_E3adaLN


class SimpleReadout(BaseReadout):
    activation_fn: str
    output: str

    @nn.compact
    def __call__(
            self,
            features: FeatureRepresentations,
            features_time: Array,
            *args,
            **kwargs
    ):

        features_nodes = features.nodes  # (num_nodes, 1, 1, num_features)
        assert features_nodes.ndim == 4, 'Features are assumed to be in the e3x convention.'
        assert features_nodes.shape[1] == 1, 'Parity must be 1.'
        assert features_nodes.shape[2] == 1, 'Maximal degree must be 0.'
        features_nodes = jnp.squeeze(features_nodes, axis=(1, 2))  # (num_nodes, num_features)

        num_features = features_nodes.shape[-1]
        num_nodes = len(features_nodes)

        features_cond = kwargs['features_cond']  # FeatureRepresentation
        if features_cond is not None:
            features_nodes_cond = features_cond.nodes
            assert features_nodes_cond.shape[1] == 1, 'Parity must be 1.'
            assert features_nodes_cond.shape[2] == 1, 'Maximal degree must be 0.'

            features_nodes_cond = jnp.squeeze(features_nodes_cond, axis=(1, 2))  # (num_nodes, num_features)

        else:
            features_nodes_cond = jnp.zeros(
                (num_nodes, num_features),
                dtype=features_time.dtype
            )  # (num_features, num_nodes)

        features_time = jnp.squeeze(features_time, axis=(1, 2))  # (num_nodes, num_features)

        c = features_time + features_nodes_cond  # (num_nodes, num_features)

        act_fn = get_activation_fn(self.activation_fn)
        shift, scale = jnp.split(
            act_fn(
                nn.Dense(
                    features=2 * num_features,
                    kernel_init=jax.nn.initializers.zeros
                )(
                    c
                )
            ),
            indices_or_sections=2,
            axis=-1
        )  # (num_nodes, num_features), (num_nodes, num_features)

        y = modulate_adaLN(
            x=nn.LayerNorm(use_bias=False, use_scale=False)(x=features_nodes),
            shift=shift,
            scale=scale
        )

        if self.output == 'drift' or self.output == 'noise':

            drift_or_noise = nn.Dense(
                features=3
            )(
                y
            )

            return drift_or_noise

        elif self.output == 'drift_and_noise':
            drift, noise = jnp.split(
                nn.Dense(
                    features=6
                )(
                    y
                ),
                indices_or_sections=2,
                axis=-1
            )

            return drift, noise  # (num_nodes, 3), (num_nodes, 3)

        else:
            raise ValueError(
                f'Unknown output option. Possible inputs are `drift`, `noise` or `drift_and_noise`. '
                f'Received {self.output=}.'
            )


class EquivariantReadout(BaseReadout):
    activation_fn: str
    output: str

    @nn.compact
    def __call__(
            self,
            features: FeatureRepresentations,
            features_time: Array,
            *args,
            **kwargs
    ):

        features_nodes = features.nodes  # (num_nodes, 1, 1, num_features)
        assert features_nodes.ndim == 4, 'Features are assumed to be in the e3x convention.'

        num_features = features_nodes.shape[-1]
        num_nodes = len(features_nodes)

        features_cond = kwargs['features_cond']  # FeatureRepresentation
        if features_cond is not None:
            features_nodes_cond = features_cond.nodes  # (num_nodes, 1, 1, num_features)
            if features_nodes_cond.shape[1:3] != (1, 1):
                raise ValueError(
                    f'Node features for conditioning must be invariant, i.e. max_degree = 0, and parity = 1.'
                    f'Received {features_nodes_cond.shape=}, '
                    f'i.e. max_degree = {get_max_degree_from_tensor_e3x(features_nodes_cond) - 1} and '
                    f'parity = {features_nodes_cond.shape[-3]}.'
                )
        else:
            features_nodes_cond = jnp.zeros_like(
                features_time,
            )  # (num_nodes, 1, 1, num_features)

        # Calculate the shift and scale parameters for adaLN and adaLN-Zero
        c = e3x.nn.add(features_nodes_cond, features_time)
        c = nn.LayerNorm()(c)  # (num_nodes, 1, 1, num_features)

        y = e3x.nn.change_max_degree_or_type(features_nodes, max_degree=1)

        # Explicitly break O(3) equivariance but features remain to be SO(3) equivariant
        y = y[:, 0, :, :] + y[:, 1, :, :]
        y = jnp.expand_dims(y, axis=-3)  # (num_nodes, 1, 4, num_features)

        act_fn = get_activation_fn(self.activation_fn)

        scale, shift = jnp.split(
            act_fn(
                nn.Dense(
                    features=3 * num_features,
                    kernel_init=jax.nn.initializers.zeros
                )(
                    c
                )
            ),
            indices_or_sections=np.array(
                [
                    2 * num_features,
                ]
            ),
            axis=-1
        )

        scale = scale.reshape(num_nodes, 1, 2, num_features)
        shift = shift.reshape(num_nodes, 1, 1, num_features)

        y = modulate_E3adaLN(
            x=EquivariantLayerNorm(use_scale=False, use_bias=False)(y),
            scale=scale,
            shift=shift
        )

        if self.output == 'noise' or self.output == 'drift':

            drift_or_noise = nn.Dense(features=1, use_bias=False)(y)  # (num_nodes, 1, 4, 1)
            drift_or_noise = jnp.squeeze(drift_or_noise, axis=(-3, -1))[:, 1:]  # (num_nodes, 3)

            return drift_or_noise

        else:

            drift_and_noise = nn.Dense(features=2, use_bias=False)(y)  # (num_nodes, 1, 4, 2)
            drift_and_noise = jnp.squeeze(drift_and_noise, axis=-3)  # (num_nodes, 4, 2)
            drift, noise = jnp.split(
                drift_and_noise, axis=-1, indices_or_sections=2
            )  # (num_nodes, 4, 1), (num_nodes, 4, 1)

            drift = jnp.squeeze(drift, axis=-1)  # (num_nodes, 4)
            noise = jnp.squeeze(noise, axis=-1)  # (num_nodes, 4)

            drift = drift[:, 1:]  # (num_nodes, 3)
            noise = noise[:, 1:]  # (num_nodes, 3)

            return drift, noise


class ClassificationReadout(nn.Module):
    num_classes: int
    num_node_features: int = 256

    @nn.compact
    def __call__(
            self,
            features: FeatureRepresentations,
            graph: jraph.GraphsTuple
    ):
        node_features = features.nodes
        num_graphs = get_number_of_graphs(graph)
        batch_segments = get_batch_segments(graph)

        # node-wise readout
        # node_features = MLP(
        #     num_layers=2,
        #     activation_fn='silu',
        #     use_bias=True,
        #     num_features=self.num_node_features
        # )(node_features)
        node_features = nn.Dense(features=self.num_node_features)(node_features)

        # node_mask = get_node_padding_mask(graph)

        # node_mask_expanded = jnp.expand_dims(
        #     node_mask,
        #     [node_features.ndim - 1 - o for o in range(0, node_features.ndim - 1)]
        # )

        # not sure if this is needed here
        # node_features = jnp.where(node_mask_expanded, node_features, 0.)  # exlude padding
        global_sum = jax.ops.segment_sum(node_features, batch_segments, num_graphs)

        # TODO: this doesn't work for equivariant features
        global_sum = global_sum.squeeze(axis=(1, 2))

        logits = nn.Dense(self.num_classes)(global_sum)
        return nn.log_softmax(logits), global_sum
