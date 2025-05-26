import jax
import optax


class EMATracker:
    def __init__(self, beta=0.999):
        """Initializes an EMA updater for model parameters.
        
        Args:
            beta: Smoothing factor (closer to 1 means slower updates).
        """
        self.beta = beta
        self.shadow_params = None  # Will be initialized with model parameters

    def initialize(self, params):
        """Initializes shadow EMA parameters to match the model parameters."""
        self.shadow_params = jax.tree.map(lambda x: x, params)

    def update(self, params):
        """Performs EMA update on the shadow parameters.
        
        Args:
            params: The current model parameters.
            
        Returns:
            Updated EMA parameters.
        """
        if self.shadow_params is None:
            self.initialize(params)
        
        # EMA update: shadow = beta * shadow + (1 - beta) * params
        self.shadow_params = optax.incremental_update(self.shadow_params, params, self.beta)
