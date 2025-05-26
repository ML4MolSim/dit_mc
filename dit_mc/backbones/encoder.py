import flax.linen as nn

from typing import Optional
from typing import Sequence

from .base import BaseLayer
from .base import BaseEdgeEmbedding
from .base import BaseNodeEmbedding
from .base import BaseEncoder
from .base import FeatureRepresentations


class EncoderModel(BaseEncoder):
    node_embedding: BaseNodeEmbedding
    layers: Sequence[BaseLayer]
    edge_embedding: Optional[BaseEdgeEmbedding] = None

    @nn.compact
    def __call__(
            self,
            graph
    ):

        features_nodes = self.node_embedding(
            graph=graph
        )  # (num_atoms, 1, 1, num_features)

        if self.edge_embedding is not None:
            features_edges = self.edge_embedding(graph=graph)
        else:
            features_edges = None

        features = FeatureRepresentations(
            nodes=features_nodes,
            edges=features_edges
        )

        for n in range(len(self.layers)):
            neural_network_layer = self.layers[n]

            features = neural_network_layer(
                graph=graph,
                features=features
            )  # (num_atoms, P, (max_degree+1)**2, num_features)

        return features
