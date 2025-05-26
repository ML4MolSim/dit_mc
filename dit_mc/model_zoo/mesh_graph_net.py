"""A simple MeshGraphNet with a classification head."""

from dit_mc.model_zoo.GeomDiT import make_graph_mesh_net_encoder
from ..backbones.forward_model import ForwardModel
from ..backbones import readout
from ..backbones import neural_network_layer


def make_mgn_encoder(
    num_mgn_layers: int,
    num_features: int,
    activation_fn: str = 'silu',
):
    encoder = make_graph_mesh_net_encoder(
        num_layers=num_mgn_layers,
        num_features=num_features,
        activation_fn=activation_fn
    )
    
    return encoder


def make_mesh_graph_net(
    num_classes: int,
    num_mgn_layers: int,
    num_dit_layers: int,
    num_features: int,
    activation_fn: str = 'silu',
):
    assert num_dit_layers == 0, "DiT layers are not supported currently."

    backbone = make_mgn_encoder(
        num_mgn_layers=num_mgn_layers,
        num_dit_layers=num_dit_layers,
        num_features=num_features,
        activation_fn=activation_fn
    )
    head = readout.ClassificationReadout(
        num_classes=num_classes
    )
    
    return ForwardModel(backbone=backbone, head=head)
