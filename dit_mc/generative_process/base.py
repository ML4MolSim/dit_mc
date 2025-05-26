
from abc import ABC


class GenerativeProcess(ABC):
    """ Generative process can be either a diffusion model, flow matching or stochastic interpolant """
    def __init__(self, net, name="GP"):
        self.net = net
        self.name = name

    def sample(self, params, graph, graph_prior, graph_cond, rng, num_steps=100):
        """ Generic sampling function for a full batch.
            Takes pretained parameters and batched data as input
        """
        pass

    def get_loss_fn(self):
        """ Get the function to compute the loss for a given data sample 
            loss_fn has signature: f(params, graph, graph_prior, graph_cond, rng)
        """
        pass
