import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np


from .base import BaseLayer
from .base import FeatureRepresentations
from .utils import get_activation_fn
from .utils import MLP
from .utils import promote_to_e3x
from .utils import get_max_degree_from_tensor_e3x
from .utils import broadcast_equivariant_multiplication
from .utils import E3MLP
from .utils import EquivariantLayerNorm
from .utils import modulate_adaLN
from .utils import modulate_E3adaLN
from ..jraph_utils import get_number_of_nodes


class MeshGraphNetLayer(BaseLayer):
    num_edge_features: int
    num_node_features: int
    activation_fn: str = 'silu'

    @nn.compact
    def __call__(
            self,
            graph: jraph.GraphsTuple,
            features: FeatureRepresentations
    ):

        # edge update
        sender_features = features.nodes[graph.senders]
        receiver_features = features.nodes[graph.receivers]
        edge_features = [
            sender_features,
            receiver_features,
            features.edges
        ]
        edge_features = MLP(
            num_layers=2,
            activation_fn=self.activation_fn,
            use_bias=True,
            num_features=self.num_edge_features
        )(
            nn.LayerNorm()(
                jnp.concatenate(edge_features, axis=-1)
            )
        )

        # node update
        num_nodes = features.nodes.shape[0]
        node_features = [
            features.nodes
        ]
        node_features.append(
            jax.ops.segment_sum(
                edge_features,
                graph.receivers,
                num_nodes
            )
        )
        node_features = MLP(
            num_layers=2,
            activation_fn=self.activation_fn,
            use_bias=True,
            num_features=self.num_node_features
        )(
            nn.LayerNorm()(
                jnp.concatenate(node_features, axis=-1)
            )
        )

        return FeatureRepresentations(
            nodes=node_features,
            edges=edge_features,
        )


class DiTLayer(BaseLayer):
    num_heads: int
    num_features_mlp: int
    activation_fn_mlp: str = 'gelu'
    activation_fn: str = 'silu'
    relative_embedding_qk_bool: bool = True
    relative_embedding_v_bool: bool = True
    act_dense_correct_bool: bool = False

    @nn.compact
    def __call__(
            self,
            graph,
            features: FeatureRepresentations,
            **kwargs
    ):
        num_nodes = get_number_of_nodes(graph)

        src_idx = graph.senders
        dst_idx = graph.receivers

        features_cond = kwargs['features_cond']  # FeatureRepresentation
        features_time = kwargs['features_time']  # (num_nodes, 1, 1, num_features)
        features_nodes = features.nodes  # (num_nodes, 1, 1, num_features)
        features_edges = features.edges  # (num_nodes, 1, 1, num_features)

        num_features = features_nodes.shape[-1]

        assert features_nodes.ndim == 4, 'Features are assumed to be in the e3x convention.'
        assert features_nodes.shape[1] == 1, 'Parity must be 1.'
        assert features_nodes.shape[2] == 1, 'Maximal degree must be 0.'

        features_nodes = jnp.squeeze(features_nodes, axis=(1, 2))
        features_time = jnp.squeeze(features_time, axis=(1, 2))

        if features_edges is not None:
            assert features_edges.shape[1] == 1, 'Parity must be 1.'
            assert features_edges.shape[2] == 1, 'Maximal degree must be 0.'

            features_edges = jnp.squeeze(features_edges, axis=(1, 2))

        if features_cond is not None:
            features_nodes_cond = features_cond.nodes
            assert features_nodes_cond.shape[1] == 1, 'Parity must be 1.'
            assert features_nodes_cond.shape[2] == 1, 'Maximal degree must be 0.'

            features_nodes_cond = jnp.squeeze(features_nodes_cond, axis=(1, 2))

            # maybe scale down features to 0s for unconditioned generation in classifier free guidance
            cond_scaling = graph.nodes['cond_scaling'] # (num_nodes)
            cond_scaling = cond_scaling[:, jnp.newaxis] # (num_nodes, 1)
            features_nodes_cond *= cond_scaling
        else:
            features_nodes_cond = jnp.zeros(
                (num_nodes, num_features),
                dtype=features_time.dtype
            )

        # Calculate the shift and scale parameters for adaLN and adaLN-Zero
        c = features_nodes_cond + features_time  # (num_nodes, num_features)
        c = nn.LayerNorm()(c)  # (num_nodes, num_features)

        act_fn = get_activation_fn(self.activation_fn)

        if self.act_dense_correct_bool:
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = jnp.split(
                nn.Dense(
                    features=6 * num_features,
                    kernel_init=jax.nn.initializers.zeros
                )(
                    act_fn(c)
                ),
                indices_or_sections=6,
                axis=-1
            )  # 6 times (num_nodes, num_features)
        else:
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = jnp.split(
                act_fn(
                    nn.Dense(
                        features=6 * num_features,
                        kernel_init=jax.nn.initializers.zeros
                    )(
                        c
                    )
                ),
                indices_or_sections=6,
                axis=-1
            )  # 6 times (num_nodes, num_features)

        features_nodes_pre_attention = modulate_adaLN(
            x=nn.LayerNorm(use_bias=False, use_scale=False)(features_nodes),
            scale=gamma1,
            shift=beta1
        )

        features_nodes_post_att = e3x.nn.SelfAttention(
            num_heads=self.num_heads,
            max_degree=0,
            use_fused_tensor=False,
            include_pseudotensors=False,
            use_relative_positional_encoding_qk=self.relative_embedding_qk_bool,
            use_relative_positional_encoding_v=self.relative_embedding_v_bool
        )(
            promote_to_e3x(features_nodes_pre_attention),
            promote_to_e3x(features_edges) if features_edges is not None else None,
            dst_idx=dst_idx,
            src_idx=src_idx,
            num_segments=num_nodes
        )  # (num_node, 1, 1, num_features)

        features_nodes_post_att = features_nodes_post_att.squeeze(
            axis=(1, 2)
        )  # (num_node, num_features)

        # Skip connection with scaling.
        features_nodes = features_nodes + features_nodes_post_att * alpha1  # (num_node, num_features)

        features_nodes_pre_mlp = modulate_adaLN(
            x=nn.LayerNorm(use_bias=False, use_scale=False)(features_nodes),
            scale=gamma2,
            shift=beta2
        )  # (num_node, num_features)

        features_nodes_post_mlp = MLP(
            num_features=[self.num_features_mlp, num_features],
            num_layers=2,
            activation_fn=self.activation_fn_mlp,
        )(
            features_nodes_pre_mlp
        )  # (num_node, num_features)

        # Skip connection with scaling.
        features_nodes = features_nodes + features_nodes_post_mlp * alpha2  # (num_node, num_features)

        return FeatureRepresentations(
            nodes=promote_to_e3x(features_nodes),
            edges=promote_to_e3x(features_edges) if features_edges is not None else None
        )


class SO3DiTLayer(BaseLayer):
    num_heads: int
    num_features_mlp: int
    max_degree: int
    include_pseudotensors: bool
    activation_fn_mlp: str = 'gelu'
    activation_fn: str = 'silu'
    act_dense_correct_bool: bool = False

    @nn.compact
    def __call__(
            self,
            graph,
            features: FeatureRepresentations,
            **kwargs
    ):
        num_nodes = get_number_of_nodes(graph)

        src_idx = graph.senders
        dst_idx = graph.receivers

        features_cond = kwargs['features_cond']  # FeatureRepresentation
        features_time = kwargs['features_time']  # (num_nodes, 1, 1, num_features)
        features_nodes = features.nodes  # (num_nodes, P, (max_degree + 1)**2, num_features)
        features_edges = features.edges  # (num_nodes, P, (max_degree + 1)**2, num_features)

        num_features = features_nodes.shape[-1]

        assert features_nodes.ndim == 4, 'Features are assumed to be in the e3x convention.'

        if features_cond is not None:
            features_nodes_cond = features_cond.nodes  # (num_nodes, 1, 1, num_features)
            if features_nodes_cond.shape[1:3] != (1, 1):
                raise ValueError(
                    f'Node features for conditioning must be invariant, i.e. max_degree = 0, and parity = 1.'
                    f'Received {features_nodes_cond.shape=}, '
                    f'i.e. max_degree = {get_max_degree_from_tensor_e3x(features_nodes_cond) - 1} and '
                    f'parity = {features_nodes_cond.shape[-3]}.'
                )
            
            # maybe scale down features to 0s for unconditioned generation in classifier free guidance
            cond_scaling = graph.nodes['cond_scaling'] # (num_nodes)
            cond_scaling = cond_scaling[:, jnp.newaxis] # (num_nodes, 1)
            cond_scaling = promote_to_e3x(cond_scaling) # (num_nodes, 1, 1, 1)
            features_nodes_cond *= cond_scaling
        else:
            features_nodes_cond = jnp.zeros_like(
                features_time,
            )  # (num_nodes, 1, 1, num_features)

        # Calculate the shift and scale parameters for adaLN and adaLN-Zero
        c = e3x.nn.add(features_nodes_cond, features_time)
        c = nn.LayerNorm()(c)

        max_degree_input = get_max_degree_from_tensor_e3x(features_nodes)

        parity_input = features_nodes.shape[-3]
        parity_output = 2 if self.include_pseudotensors else 1

        act_fn = get_activation_fn(self.activation_fn)

        if self.act_dense_correct_bool:
            c = nn.Dense(
                    features=num_features * (max_degree_input + 1) * parity_input + num_features * (
                        self.max_degree + 1) * parity_output + 2 * num_features + 2 * num_features * (
                        self.max_degree + 1) * parity_output,
                    kernel_init=jax.nn.initializers.zeros
            )(
                act_fn(c)
            )  # 6 times (num_nodes, num_features)
        else:
            c = act_fn(
                nn.Dense(
                    features=num_features * (max_degree_input + 1) * parity_input + num_features * (
                        self.max_degree + 1) * parity_output + 2 * num_features + 2 * num_features * (
                        self.max_degree + 1) * parity_output,
                    kernel_init=jax.nn.initializers.zeros
                )(
                    c
                )
            )  # 6 times (num_nodes, num_features)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = jnp.split(
            c,
            indices_or_sections=np.array(
                [
                    num_features * (max_degree_input + 1) * parity_input,
                    num_features * (max_degree_input + 1) * parity_input + num_features,
                    num_features * (max_degree_input + 1) * parity_input + num_features + num_features * (
                                self.max_degree + 1) * parity_output,
                    num_features * (max_degree_input + 1) * parity_input + num_features + num_features * (
                            self.max_degree + 1) * parity_output * 2,
                    num_features * (max_degree_input + 1) * parity_input + num_features + num_features * (
                            self.max_degree + 1) * parity_output * 2 + num_features
                    # num_features * (max_degree_input + 1) * parity_input + num_features + num_features * (
                    #         max_degree_input + 1) * parity_output + num_features + num_features * (
                    #             max_degree_input + 1) * parity_output
                ]
            ),
            axis=-1
        )

        gamma1 = gamma1.reshape(num_nodes, parity_input, (max_degree_input + 1), num_features)
        beta1 = beta1.reshape(num_nodes, 1, 1, num_features)
        alpha1 = alpha1.reshape(num_nodes, parity_output, (self.max_degree + 1), num_features)
        gamma2 = gamma2.reshape(num_nodes, parity_output, (self.max_degree + 1), num_features)
        beta2 = beta2.reshape(num_nodes, 1, 1, num_features)
        alpha2 = alpha2.reshape(num_nodes, parity_output, (self.max_degree + 1), num_features)

        # SO3 equivariant adaLN.
        features_nodes_pre_attention = modulate_E3adaLN(
            x=EquivariantLayerNorm(use_bias=False, use_scale=False)(features_nodes),
            scale=gamma1,
            shift=beta1
        )

        features_nodes_post_att = e3x.nn.SelfAttention(
            num_heads=self.num_heads,
            max_degree=self.max_degree,
            use_fused_tensor=False,
            include_pseudotensors=self.include_pseudotensors,
        )(
            features_nodes_pre_attention,
            features_edges,
            dst_idx=dst_idx,
            src_idx=src_idx,
            num_segments=num_nodes
        )  # (num_nodes, parity, (max_degree + 1)**2, num_features)

        # Skip connection with SO3 equivariant per-degree / parity scaling.
        features_nodes = e3x.nn.add(
            features_nodes,
            broadcast_equivariant_multiplication(
                factor=alpha1,
                tensor=features_nodes_post_att
            )
        )  # (num_nodes, parity, (max_degree + 1)**2, num_features)

        features_nodes_pre_mlp = modulate_E3adaLN(
            x=EquivariantLayerNorm(use_bias=False, use_scale=False)(features_nodes),
            scale=gamma2,
            shift=beta2
        )  # (num_nodes, parity, (max_degree + 1)**2, num_features)

        features_nodes_post_mlp = E3MLP(
            num_features=[self.num_features_mlp, num_features],
            num_layers=2,
            activation_fn=self.activation_fn_mlp,
        )(
            features_nodes_pre_mlp
        )  # (num_nodes, parity, (max_degree + 1)**2, num_features)

        # Skip connection with scaling.
        features_nodes = e3x.nn.add(
            features_nodes,
            broadcast_equivariant_multiplication(
                factor=alpha2,
                tensor=features_nodes_post_mlp
            )
        )   # (num_nodes, parity, (max_degree + 1)**2, num_features)

        return FeatureRepresentations(
            nodes=features_nodes,
            edges=features_edges
        )
