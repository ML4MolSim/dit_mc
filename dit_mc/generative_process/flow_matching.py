import jax
import jraph
import jax.numpy as jnp

from dit_mc.generative_process.base import GenerativeProcess
from dit_mc.jraph_utils import (
    compute_batch_statistics,
    duplicate_graph, 
    move_graph_positions,
)
from dit_mc.training.utils import aggregate_node_error, kabsch_align


def center_data(x, batch_segments, num_graphs):
  x_mean_per_graph = jraph.segment_mean(x, batch_segments, num_graphs)  # (num_graphs, ...)
  x = x - x_mean_per_graph[batch_segments]  # (num_nodes, ...)
  return x


def sample_mixture_beta_uniform(key, shape, p1=1.9, p2=1.0, uniform_prob=0.02):
    rng_beta, rng_uniform, rng_choice = jax.random.split(key, 3)

    samples_beta = jax.random.beta(rng_beta, p1, p2, shape=shape)
    samples_uniform = jax.random.uniform(rng_uniform, shape=shape)

    u = jax.random.uniform(rng_choice, shape=shape)
    return jnp.where(u < uniform_prob, samples_uniform, samples_beta)


class FlowMatching(GenerativeProcess):
    def __init__(
            self, 
            prior, 
            sigma, 
            align_bool=True, 
            conditioning_bool=False, 
            regress_x1_bool=True,
            weighted_loss_bool=True,
            mixture_tau_bool=False,
            free_guidance_bool=False,
            free_guidance_prob=0.1,
            self_conditioning_bool=False,
            self_conditioning_prob=0.5,
            **kwargs):
        
        super().__init__(**kwargs)
        
        self.prior = prior
        self.sigma = sigma
        self.align_bool = align_bool
        self.conditioning_bool = conditioning_bool
        self.regress_x1_bool = regress_x1_bool
        self.weighted_loss_bool = weighted_loss_bool

        self.mixture_tau_bool = mixture_tau_bool

        self.free_guidance_bool = free_guidance_bool
        self.free_guidance_prob = free_guidance_prob

        self.self_conditioning_bool = self_conditioning_bool
        self.self_conditioning_prob = self_conditioning_prob 

        # use this for sampling
        self.fwd_jit = jax.jit(self.forward)

    def forward(self, params, graph_latent, graph_cond, time_latent):
        nn_out, _ = self.net.apply(
            params,
            time_latent=time_latent,
            graph_latent=graph_latent,
            graph_cond=graph_cond if self.conditioning_bool else None,
        ) # (num_nodes, ...)
        return nn_out

    def clean_prediction(self, nn_out, batch, graph_latent, time_latent):
        nn_out = center_data(nn_out, batch.batch_segments, batch.num_graphs)
        nn_out = jnp.where(batch.node_mask_expanded, nn_out, 0.)

        if self.regress_x1_bool:
            x1_pred = nn_out
            vt = jnp.einsum('t, tn... -> tn...', 1 / (1. - time_latent), nn_out - graph_latent.nodes['positions'])
        else:
            x1_pred = graph_latent.nodes['positions'] + jnp.einsum('t, tn... -> tn...', 1. - time_latent, nn_out)
            vt = nn_out
        return x1_pred, vt

    def sample(
            self, 
            params, 
            graph, 
            graph_prior, 
            graph_cond, 
            rng, 
            num_steps=50,
            free_guidance_scale=1.0,
            logarithmic_time_bool=False,
            return_trajectory=False,
    ):
        rng, rng_prior = jax.random.split(rng, num=2)

        batch = compute_batch_statistics(graph)

        x1 = graph.nodes['x1']

        xtau = self.prior.sample(
            rng_prior,
            shape=x1.shape,
            graph_prior=graph_prior,
        )
        xtau = center_data(xtau, batch.batch_segments, batch.num_graphs)
        xtau = jnp.where(batch.node_mask_expanded, xtau, 0.)  # (num_nodes, ...)

        # Create graphs
        graph_latent = duplicate_graph(graph)
        graph_latent.nodes['positions'] = xtau
        graph_latent.nodes['cond_scaling'] = jnp.ones(shape=(len(x1),))
        graph_latent.edges['cond_scaling'] = jnp.ones(shape=len(graph.senders,))

        if self.self_conditioning_bool:
            graph_latent.nodes['self_cond'] = jnp.zeros_like(x1)

        if logarithmic_time_bool:
            # spend more time in parts of the vector field closer to t = 1
            base = 10
            taus = jnp.logspace(0.0, 1.0, num_steps + 1, base=base)
            taus = (taus - 1) / (base - 1)
            taus = (1 - taus)[::-1]
        else:
            taus = jnp.linspace(0.0, 1.0, num_steps + 1)

        ones = jnp.ones(shape=(batch.num_graphs,))
        ones = jnp.take(ones, batch.batch_segments)
        ones = jnp.where(batch.node_mask, ones, 0.)

        xtaus = [xtau]
        for n, tau in enumerate(taus[:-1]):

            dtau = taus[n + 1] - taus[n]
            time_latent = ones * tau

            if n > 0 and self.self_conditioning_bool:
                graph_latent = duplicate_graph(graph_latent)
                graph_latent.nodes['self_cond'] = x1_pred

            nn_out = self.fwd_jit(
                params, 
                time_latent=time_latent,
                graph_latent=graph_latent,
                graph_cond=graph_cond,
            )
            
            x1_pred, vt = self.clean_prediction(
                nn_out,
                batch=batch,
                time_latent=time_latent,
                graph_latent=graph_latent,
            )

            if self.free_guidance_bool:
                graph_latent = duplicate_graph(graph_latent)
                graph_latent.nodes['cond_scaling'] = jnp.zeros(shape=(len(x1),))
                graph_latent.edges['cond_scaling'] = jnp.zeros(shape=len(graph.senders,))

                nn_out_uncond = self.fwd_jit(
                    params, 
                    time_latent=time_latent,
                    graph_latent=graph_latent,
                    graph_cond=graph_cond,
                )
            
                x1_pred_uncond, vt_uncond = self.clean_prediction(
                    nn_out_uncond,
                    batch=batch,
                    time_latent=time_latent,
                    graph_latent=graph_latent,
                )

                x1_pred = x1_pred_uncond + free_guidance_scale * (x1_pred - x1_pred_uncond)
                vt = vt_uncond + free_guidance_scale * (vt - vt_uncond)

                graph_latent = duplicate_graph(graph_latent)
                graph_latent.nodes['cond_scaling'] = jnp.ones(shape=(len(x1),))
                graph_latent.edges['cond_scaling'] = jnp.ones(shape=len(graph.senders,))

            dx = dtau * vt
            graph_latent = move_graph_positions(
                graph_latent,
                delta_positions=dx
            )

            xtaus.append(graph_latent.nodes['positions'])

        if return_trajectory:
            return graph_latent, xtaus
        else:
            return graph_latent

    def get_loss_fn(self):
        def loss_fn(params, graph, graph_prior, graph_cond, rng):
            rng, rng_x0, rng_tau, rng_z = jax.random.split(rng, num=4)

            batch = compute_batch_statistics(graph)

            x1 = graph.nodes['x1']  # (num_nodes, ...)
            x1 = center_data(x1, batch.batch_segments, batch.num_graphs)

            x0 = self.prior.sample(
                rng_x0,
                shape=x1.shape,
                graph_prior=graph_prior,
            )  # (num_nodes, ...)
            x0 = center_data(x0, batch.batch_segments, batch.num_graphs)
            x0 = jnp.where(batch.node_mask_expanded, x0, 0.)  # (num_nodes, ...)

            if self.align_bool:
                x0 = kabsch_align(graph, x0, x1) # (num_nodes, ...)
                x0 = center_data(x0, batch.batch_segments, batch.num_graphs)
                x0 = jnp.where(batch.node_mask_expanded, x0, 0.)  # (num_nodes, ...)

            if self.mixture_tau_bool:
                tau_per_graph = sample_mixture_beta_uniform(
                    rng_tau,
                    shape=(batch.num_graphs,)
                )  # (num_graphs)
            else:
                tau_per_graph = jax.random.uniform(
                    rng_tau,
                    shape=(batch.num_graphs,)
                )  # (num_graphs)

            tau = jnp.take(tau_per_graph, batch.batch_segments)  # (num_nodes)
            tau = jnp.where(batch.node_mask, tau, 0.)  # (num_nodes)

            z = jax.random.normal(
                rng_z,
                shape=x1.shape,
            )  # (num_nodes, ...)
            z = center_data(z, batch.batch_segments, batch.num_graphs)
            z = jnp.where(batch.node_mask_expanded, z, 0.)  # (num_nodes, ...)

            # Coefficients
            alpha = 1. - tau  # (num_nodes)
            beta = tau  # (num_nodes)
            if self.regress_x1_bool:
                gamma = self.sigma * jnp.ones_like(tau)
            else:
                gamma = self.sigma * jnp.sqrt(tau * (1 - tau))

            # interpolate
            A = jnp.einsum('t, tn... -> tn...', alpha, x0) # A[t, :] = alpha[t] * x0[t, :]
            B = jnp.einsum('t, tn... -> tn...', beta, x1)  # B[t, :] = beta[t] * x1[t, :]
            C = jnp.einsum('t, tn... -> tn...', gamma, z)  # C[t, :] = gamma[t] * z[t, :]
            xtau = A + B + C  # (num_nodes, ...)
            xtau = center_data(xtau, batch.batch_segments, batch.num_graphs)
            xtau = jnp.where(batch.node_mask_expanded, xtau, 0.)  # (num_nodes, ...)

            # Create graphs
            graph_latent = duplicate_graph(graph)
            graph_latent.nodes['positions'] = xtau
            graph_latent.nodes['cond_scaling'] = jnp.ones(shape=(len(x1),))
            graph_latent.edges['cond_scaling'] = jnp.ones(shape=len(graph.senders,))
           
            if self.free_guidance_bool:
                def get_ones(_):
                    return jnp.ones(shape=(1,))
                
                def get_zeros(_):
                    return jnp.zeros(shape=(1,))
                
                rng, rng_fg = jax.random.split(rng)
                
                cond_scaling = jax.lax.cond(
                    jax.random.uniform(rng_fg, shape=(1,))[0] < self.free_guidance_prob,
                    get_zeros,
                    get_ones,
                    None
                )

                graph_latent.nodes['cond_scaling'] = graph_latent.nodes['cond_scaling'] * cond_scaling
                graph_latent.edges['cond_scaling'] = graph_latent.edges['cond_scaling'] * cond_scaling

            if self.self_conditioning_bool:
                graph_latent = duplicate_graph(graph_latent)
                graph_latent.nodes['self_cond'] = jnp.zeros_like(x1)

                def get_x1_pred(_):
                    nn_out = self.forward(
                        params, 
                        time_latent=tau, 
                        graph_latent=graph_latent, 
                        graph_cond=graph_cond,
                    )
                    
                    x1_pred, _ = self.clean_prediction(
                        nn_out,
                        batch=batch,
                        time_latent=tau,
                        graph_latent=graph_latent,
                    )

                    return jax.lax.stop_gradient(x1_pred)

                def get_zeros(_):
                    return jnp.zeros_like(x1)

                rng, rng_sc = jax.random.split(rng)

                x1_pred = jax.lax.cond(
                    jax.random.uniform(rng_sc, shape=(1,))[0] < self.self_conditioning_prob,
                    get_x1_pred,
                    get_zeros,
                    None
                )

                graph_latent = duplicate_graph(graph_latent)
                graph_latent.nodes['self_cond'] = x1_pred

            nn_out = self.forward(
                params,
                time_latent=tau,
                graph_latent=graph_latent,
                graph_cond=graph_cond
            )  # (num_nodes, ...)

            x1_pred, vt = self.clean_prediction(
                nn_out,
                batch=batch,
                time_latent=tau,
                graph_latent=graph_latent,
            )

            if self.regress_x1_bool:
                ut_hat = x1_pred

                ut = x1
                ut = jnp.where(batch.node_mask_expanded, ut, 0.)
            else:
                I_grad = x1 - x0   # (num_nodes, ...)
                I_grad = jnp.where(batch.node_mask_expanded, I_grad, 0.)  # (num_nodes, ...)

                gamma_grad = 0.5 * self.sigma * (1. - 2 * tau) / jnp.sqrt(tau * (1. - tau))  # (num_nodes)
                C_grad = jnp.einsum('t, tn... -> tn...', gamma_grad, z)  # (num_nodes, ...)
                C_grad = jnp.where(batch.node_mask_expanded, C_grad, 0.)  # (num_nodes, ...)

                ut_hat = vt

                ut = I_grad + C_grad # (num_nodes, ...)
                ut = jnp.where(batch.node_mask_expanded, ut, 0.)

            per_node_error = (ut_hat - ut) ** 2 # (num_nodes)
            # per_node_error = jnp.sum(jnp.square(vt - ut), axis=-1) # (num_nodes)
            # per_node_error = jnp.linalg.norm(vt - ut, ord=2, axis=-1) # (num_nodes)

            graph_weight = 1.
            if self.weighted_loss_bool:
                # clip tau at 0.9 similar to Yim et al. in their work
                # Fast protein backbone generation with SE(3) flow matching
                graph_weight = 1.0 / ((1.0 - jnp.minimum(tau_per_graph, 0.9)) ** 2)

            loss = aggregate_node_error(
                node_error=per_node_error,
                batch_segments=batch.batch_segments,
                graph_mask=batch.graph_mask,
                graph_weight=graph_weight,
                scale = 1.
            )

            return loss
        return loss_fn
