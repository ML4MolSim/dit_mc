import abc
import flax.linen as nn
import jraph

from collections import namedtuple
from jaxtyping import Array
from typing import Optional


GenerativeLayer = namedtuple(
    'GenerativeLayer', field_names=("merger", "encoder")
)

FeatureRepresentations = namedtuple(
    'FeatureRepresentations', field_names=('nodes', 'edges')
)


class BaseLayer(nn.Module):
    @abc.abstractmethod
    def __call__(
            self,
            features: FeatureRepresentations,
            graph: Optional[jraph.GraphsTuple],
            **kwargs
    ) -> FeatureRepresentations:
        pass

# class BaseEncoder(nn.Module):
#     @abc.abstractmethod
#     def __call__(
#             self,
#             graph,
#             ztau: Optional[Array] = None,
#             tau: Optional[Array] = None,
#             node_features_cond: Optional[Array] = None
#     ):
#         """
#           Forward function of encoder blocks. They can be used either as a standalone neural network or as submodule in
#           a GeneralNet. They can be used as pure conditioner or as a model processing the current state of the latent
#           variables as well as the current time step.
#
#         Args:
#             graph (): jraph.GraphsTuple, which stores the relevant graph properties of the original graph,
#                 i.e. connectivity (senders and receivers), atomic types, positions, ...
#             ztau (): Current latent state. Typically, atomic positions or displacements, (num_nodes, 3)
#             tau (): Current time step, associated with the latent state, (num_nodes)
#
#         Returns:
#
#         """
#         pass
#
#     @abc.abstractmethod
#     def is_equivariant(self) -> bool:
#         raise NotImplementedError(
#             'Implement function which specifies whether this is an equivariant architecture, or not.'
#         )
#
#     def get_cutoff(self) -> float:
#         raise NotImplementedError(
#             'Implement function which returns the cutoff.'
#         )


class BaseTimeEmbedding(nn.Module):
    def __call__(
            self,
            time_latent: Array
    ):
        """

        Args:
            time_latent (): The latent times, (num_nodes)

        Returns:

        """
        pass


class BaseNodeEmbedding(nn.Module):
    def __call__(
            self,
            graph: jraph.GraphsTuple
    ):
        pass


class BaseEdgeEmbedding(nn.Module):
    def __call__(
            self,
            graph: jraph.GraphsTuple
    ):
        pass


class BaseMerger(nn.Module):
    @abc.abstractmethod
    def __call__(
            self,
            features: FeatureRepresentations,
            **kwargs
    ):
        pass


class BaseReadout(nn.Module):
    @abc.abstractmethod
    def __call__(
            self,
            features,
            *args,
            **kwargs
    ):
        pass


class BaseEncoder(nn.Module):
    @abc.abstractmethod
    def __call__(
            self,
            graph
    ) -> Array:

        pass
