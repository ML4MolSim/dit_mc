import flax.linen as nn
import jraph
from .base import BaseEncoder


class ForwardModel(nn.Module):
    backbone: BaseEncoder
    head: nn.Module

    @nn.compact
    def __call__(
            self,
            graph: jraph.GraphsTuple
    ):
        features = self.backbone(graph)
        return self.head(features, graph)
