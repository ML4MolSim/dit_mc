import flax.linen as nn
import jraph

from jaxtyping import Array
from typing import Optional
from typing import Sequence

from .base import BaseReadout
from .base import GenerativeLayer
from .base import BaseEdgeEmbedding
from .base import BaseNodeEmbedding
from .base import BaseTimeEmbedding
from .base import BaseEncoder
from .base import FeatureRepresentations


class GenerativeModel(nn.Module):
    node_embedding: BaseNodeEmbedding
    time_embedding: BaseTimeEmbedding
    layers: Sequence[GenerativeLayer]
    readout: BaseReadout
    edge_embedding: Optional[BaseEdgeEmbedding] = None
    conditioner: Optional[BaseEncoder] = None
    conditioning_bool: bool = False
    name: str

    def get_output(self):
        return self.readout.output

    def setup(self):
        allowed_outputs = [
            'noise',
            'drift',
            'drift_and_noise'
        ]

        if self.get_output() not in allowed_outputs:
            raise ValueError(
                f"`output` must be one of {allowed_outputs}. "
                f"received {self.output}"
            )

    @nn.compact
    def __call__(
            self,
            time_latent: Array,
            graph_latent: jraph.GraphsTuple,
            graph_cond: Optional[jraph.GraphsTuple] = None,

    ):
        if self.conditioning_bool is True:
            features_cond = self.conditioner(
                graph_cond
            )
        else:
            features_cond = None

        features_nodes = self.node_embedding(
            graph=graph_latent
        )  # (num_atoms, *dims)

        features_time = self.time_embedding(
            time_latent=time_latent,
        )  # (num_atoms)

        if self.edge_embedding is not None:
            features_edges = self.edge_embedding(graph=graph_latent)
        else:
            features_edges = None

        features = FeatureRepresentations(
            nodes=features_nodes, edges=features_edges
        )

        for n in range(len(self.layers)):
            merger = self.layers[n].merger
            encoder = self.layers[n].encoder

            features = merger(
                features=features,
                features_cond=features_cond,
                features_time=features_time,
                graph_latent=graph_latent,
                graph_cond=graph_cond
            )

            features = encoder(
                graph=graph_latent,
                features=features,
                features_cond=features_cond,
                features_time=features_time,
                features_edges=features_edges,
                graph_cond=graph_cond
            )

        return self.readout(
            features,
            features_time=features_time,
            features_cond=features_cond
        )
