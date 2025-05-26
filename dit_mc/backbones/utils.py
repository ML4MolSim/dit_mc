import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import math
from functools import partial
from jax import lax
from typing import Callable, Sequence, Union

from .base import FeatureRepresentations
from ..jraph_utils import make_dummy_graph


def safe_mask(mask, fn: Callable, operand: jnp.ndarray, placeholder: float = 0.) -> jnp.ndarray:
    """
    Safe mask which ensures that gradients flow nicely. See also
    https://github.com/google/jax-md/blob/b4bce7ab9b37b6b9b2d0a5f02c143aeeb4e2a560/jax_md/util.py#L67

    Args:
        mask (array_like): Array of booleans.
        fn (Callable): The function to apply at entries where mask=True.
        operand (array_like): The values to apply fn to.
        placeholder (int): The values to fill in if mask=False.

    Returns: New array with values either being the output of fn or the placeholder value.

    """
    masked = jnp.where(mask, operand, 0)
    return jnp.where(mask, fn(masked), placeholder)


def modulate_adaLN(x, shift, scale):
    """

    Args:
        x (): Features to be modulated. (num_atoms, num_features)
        shift (): Shift to be added. (num_atoms, num_features)
        scale (): Scale to be multiplied. (num_atoms, num_features)

    Returns:

    """

    if not x.shape == scale.shape == shift.shape:
        raise ValueError(
            f'Shape of features, scale and shift must be identical. '
            f'Received {x.shape=}, {scale.shape=} and {shift.shape=}.'
        )

    return x * (1 + scale) + shift


def modulate_E3adaLN(x, shift, scale):
    """

    Args:
        x (): Features to be modulated. (num_atoms, parity, (max_degree + 1)**2, num_features)
        shift (): Shift to be added. (num_atoms, 1, 1, num_features)
        scale (): Scale to be multiplied. (num_atoms, num_features)

    Returns:

    """

    x_scaled = broadcast_equivariant_multiplication(
        factor=1 + scale,
        tensor=x
    )

    return e3x.nn.add(
        x_scaled,
        shift
    )


def cumsum(x: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (1, 0)
    x = jnp.pad(x, pad_width, mode='constant', constant_values=0)
    return jnp.cumsum(x, axis=axis)


def get_pos_indices(n_node, num_nodes):
    return jnp.arange(num_nodes) - jnp.repeat(cumsum(n_node)[:-1], n_node, total_repeat_length=num_nodes)


def get_index_embedding(indices, emb_dim, max_len=256):
    """Creates sine / cosine positional embeddings from prespecified indices.

    Args:
        indices: offsets of size [num_tokens,] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum length

    Returns:
        positional embedding of shape [num_tokens, emb_dim]
    """
    K = jnp.arange(emb_dim // 2)
    pos_embedding_sin = jnp.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K / emb_dim))
    )
    pos_embedding_cos = jnp.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K / emb_dim))
    )
    pos_embedding = jnp.concatenate([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class GaussianRandomFourierFeatures(nn.Module):
    features: int
    sigma: float = 1.
    dtype = jnp.float32
    param_dtype = jnp.float32

    @nn.compact
    def __call__(
            self,
            x  # (..., d)
    ):
        if self.features % 2 != 0:
            raise ValueError(
                f'features must be even. '
                f'received {self.features=}'
            )

        b = self.param(
            'b',
            jax.nn.initializers.normal(self.sigma),
            (x.shape[-1], self.features // 2),
            self.param_dtype
        )  # (d, features // 2)

        bT_x = jnp.einsum(
            '...d, dh -> ...h',
            x,
            b
        )  # (..., features // 2)

        cos = jnp.cos(2 * jnp.pi * bT_x)  # (..., features // 2)
        sin = jnp.sin(2 * jnp.pi * bT_x)  # (..., features // 2)

        # gamma contains alternating cos and sin terms by first stacking and then reshaping.
        gamma = jnp.stack(
            [cos, sin],
            axis=-1
        ).reshape(*cos.shape[:-1], -1)  # (..., features)

        return gamma


def get_activation_fn(name: str):
    if name == 'identity':
        return lambda x: x
    else:
        return getattr(
            jax.nn,
            name
        )


def get_e3x_activation_fn(name: str):
    if name == 'identity':
        return lambda x: x
    else:
        return getattr(
            e3x.nn,
            name
        )


class MLP(nn.Module):
    num_layers: int = 2
    activation_fn: str = 'identity'
    num_features: Union[int, Sequence[int]] = None
    use_bias: bool = True
    output_is_zero_at_init: bool = False

    @nn.compact
    def __call__(self, x):

        activation_fn = get_activation_fn(
            name=self.activation_fn
        )

        if type(self.num_features) == list or type(self.num_features) == tuple:
            num_features = self.num_features
        else:
            num_features = x.shape[-1] if self.num_features is None else self.num_features
            num_features = [num_features] * self.num_layers

        for n in range(self.num_layers):

            if self.output_is_zero_at_init is True and n == self.num_layers - 1:
                # Initialize the last layer with zeros such that the output of the MLP
                kernel_init = jax.nn.initializers.zeros
            else:
                kernel_init = jax.nn.initializers.lecun_normal()

            x = nn.Dense(
                num_features[n],
                use_bias=self.use_bias,
                kernel_init=kernel_init
            )(
                x
            )

            # do not apply activation in the last layer.
            if n < self.num_layers - 1:
                x = activation_fn(
                    x
                )

        return x


class E3MLP(nn.Module):
    num_layers: int = 2
    activation_fn: str = 'identity'
    num_features: Union[int, Sequence[int]] = None
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):

        if type(self.num_features) == list:
            num_features = self.num_features
        else:
            num_features = x.shape[-1] if self.num_features is None else self.num_features
            num_features = [num_features] * self.num_layers

        activation_fn = get_e3x_activation_fn(
            name=self.activation_fn
        )

        num_features = x.shape[-1] if self.num_features is None else self.num_features

        for n in range(self.num_layers):
            x = e3x.nn.Dense(
                num_features[n],
                use_bias=self.use_bias
            )(
                x
            )

            # do not apply activation in the last layer.
            if n < self.num_layers - 1:
                x = activation_fn(
                    x
                )

        return x


def safe_norm(x, axis=0, keepdims=False) -> jnp.ndarray:
    """Take gradient safe norm.

  Args:
    x: Tensor.
    axis: Axis along which norm is taken.
    keepdims: If dimension should be kept.

  Returns: Tensor.

  """
    u = (x ** 2).sum(axis=axis, keepdims=keepdims)
    return safe_mask(mask=u > 0, fn=jnp.sqrt, operand=u, placeholder=0.0)


def promote_to_e3x(x: jnp.ndarray) -> jnp.ndarray:
    """
  Promote an invariant node representation to a tensor that matches the shape
  convention of e3x, i.e. adding an axis for parity and irreps.

    Args:
      x: Tensor of shape (n, F)

    Returns: Tensor of shape (n, 1, 1, F)
  """
    assert x.ndim == 2
    return x[:, None, None, :]


def make_degree_repeat_fn(degrees: Sequence[int], axis: int = -1):
    repeats = np.array([2 * y + 1 for y in degrees])
    repeat_fn = partial(np.repeat, repeats=repeats, axis=axis)
    return repeat_fn


class EquivariantLayerNorm(nn.Module):
    use_scale: bool = True
    use_bias: bool = True

    bias_init: Callable = nn.initializers.zeros
    scale_init: Callable = nn.initializers.ones

    epsilon: float = 1e-6
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        """
            x.shape: (N, 1 or 2, (max_degree + 1)^2, features)
        """
        assert x.ndim == 4

        max_degree = int(np.rint(np.sqrt(x.shape[-2]))) - 1
        num_features = x.shape[-1]
        num_atoms = x.shape[-4]

        has_pseudotensors = x.shape[-3] == 2
        has_ylms = x.shape[-2] > 1

        if has_pseudotensors or has_ylms:
            plm_axes = x.shape[-3:-1]

            y = x.reshape(num_atoms, -1, num_features)  # (N, plm, features)
            y00, ylm = jnp.split(
                y,
                axis=1,
                indices_or_sections=np.array([1])
            )  # (N, 1, features), (N, plm - 1, features)

            # Construct the segment sum indices for summing over degree and parity channels.
            repeat_fn_even = make_degree_repeat_fn(degrees=list(range(1, max_degree + 1)))
            sum_idx_even = repeat_fn_even(np.arange(max_degree))

            if has_pseudotensors:
                repeat_fn_odd = make_degree_repeat_fn(degrees=list(range(max_degree + 1)))
                sum_idx_odd = repeat_fn_odd(np.arange(max_degree, 2 * max_degree + 1))
            else:
                sum_idx_odd = np.array([], dtype=sum_idx_even.dtype)

            sum_idx = np.concatenate([sum_idx_even, sum_idx_odd], axis=0)

            ylm_sum_squared = jax.vmap(
                partial(
                    jax.ops.segment_sum,
                    segment_ids=sum_idx,
                    num_segments=2 * max_degree + 1 if has_pseudotensors else max_degree
                )
            )(
                lax.square(ylm),
            )  # (N, parity * max_degree + 1 or max_degree, features)

            ylm_inv = safe_mask(
                ylm_sum_squared > self.epsilon,
                lax.sqrt,
                ylm_sum_squared
            )

            _, var_lm = nn.normalization._compute_stats(
                ylm_inv,
                axes=-1,
                dtype=None
            )  # (N, parity * max_degree + 1 or max_degree)

            mul_lm = lax.rsqrt(var_lm + jnp.asarray(self.epsilon, dtype=var_lm.dtype))
            # (N, parity * max_degree + 1 or max_degree)

            if self.use_scale:
                scales_lm = self.param(
                    'scales_lm',
                    self.scale_init,
                    (var_lm.shape[-1], ),
                    self.param_dtype
                )  # (parity * max_degree + 1 or max_degree)

                mul_lm = mul_lm * scales_lm  # (N, parity * max_degree + 1 or max_degree)

            mul_lm = jnp.expand_dims(mul_lm, axis=-1)  # (N, parity * max_degree + 1 or max_degree, 1)

            ylm = ylm * mul_lm[:, sum_idx, :]  # (N, plm - 1, features)

            y00 = nn.LayerNorm(
                use_scale=self.use_scale,
                use_bias=self.use_bias,
                scale_init=self.scale_init,
                bias_init=self.bias_init
            )(y00)  # (N, 1, features)

            y = jnp.concatenate([y00, ylm], axis=1)  # (N, plm, features)
            return y.reshape(num_atoms, *plm_axes, num_features)  # (N, 1 or 2, (max_degree + 1)^2, features)
        else:
            return nn.LayerNorm(
                use_scale=self.use_scale,
                use_bias=self.use_bias,
                scale_init=self.scale_init,
                bias_init=self.bias_init
            )(x)  # (N, 1, features)


def get_number_of_params(x):
    """
    Get number of params. Can be either GeneralNet or just a PyTree of params.

    Args:
        x (): GeneralNet or a PyTree of params.

    Returns:
        Number of params.

    """

    if isinstance(x, nn.Module):
        graph = make_dummy_graph(
            num_atoms=7
        )

        latent_time = np.ones(
            (7,)
        )
        latent_state = np.ones(
            (7, 3)
        )

        params = x.init(
            jax.random.PRNGKey(0),
            graph,
            latent_time=latent_time,
            latent_state=latent_state
        )

    else:
        params = x

    param_count = sum(
        x.size for x in jax.tree_util.tree_leaves(params)
    )

    return param_count


def equivariant_concatenation(x, y):
    assert x.ndim == 4
    assert y.ndim == 4

    x_shape = x.shape
    y_shape = y.shape

    x_invariant_bool = x_shape[1:3] == (1, 1)
    y_invariant_bool = y_shape[1:3] == (1, 1)

    # Easy case of all dimensions equal.
    if x_shape[:3] == y_shape[:3]:
        return jnp.concatenate([x, y], axis=-1)

    # One of the the two is invariant.
    elif x_invariant_bool is True or y_invariant_bool is True:
        if x_invariant_bool:
            x = e3x.nn.add(x, jnp.zeros((*y.shape[:3], x.shape[-1])))  # hack to bring to correct shape

            # does the same as this code:
            # _x = jnp.zeros_like(y)
            # x = _x.at[:, 0, 0, :].set(jnp.squeeze(x, axis=(1, 2)))

            return jnp.concatenate([x, y], axis=-1)
        else:
            y = e3x.nn.add(y, jnp.zeros((*x.shape[:3], y.shape[-1])))  # hack to bring to correct shape

            # does the same as this code:
            # _y = jnp.zeros_like(x)
            # y = _y.at[:, 0, 0, :].set(jnp.squeeze(y, axis=(1, 2)))

            return jnp.concatenate([x, y], axis=-1)
    else:
        raise NotImplementedError(
            f'At the moment, equivariant concatenation is only supported for both features having same '
            f'max_degree and parity or one of the features invariant (P=1, max_degree=0) and the other arbitrary. '
            f'received {x.shape} and {y.shape}.'
        )


def get_max_degree_from_tensor_e3x(x):
    return int(np.rint(np.sqrt(x.shape[-2]) - 1).item())


def broadcast_equivariant_multiplication(factor, tensor):
    max_degree_tensor = get_max_degree_from_tensor_e3x(tensor)
    max_degree_factor = factor.shape[-2] - 1

    assert factor.shape[-1] == tensor.shape[-1], \
        f'Feature dimensions must align. Received {factor.shape=} and {tensor.shape=}'

    assert len(tensor) == len(factor), \
        f'Leading axis must align. Received {len(factor)=} and {len(tensor)=}'

    assert max_degree_factor == max_degree_tensor, \
        f'Max degree must align. Received {max_degree_factor=} and {max_degree_tensor=}'

    repeats_factor = [2 * ell + 1 for ell in range(max_degree_factor + 1)]

    return jnp.repeat(
        factor,
        axis=-2,
        repeats=np.array(repeats_factor),
        total_repeat_length=(max_degree_tensor + 1) * (max_degree_tensor + 1)
    ) * tensor
