from typing import Any
import wandb
from tqdm import tqdm
import jax
import jax.random as random
import numpy as np
import optax
from flax.training import train_state
from flax.training import orbax_utils
from flax import core, struct

from orbax import checkpoint
from dit_mc.generative_process.base import GenerativeProcess
from dit_mc.training.ema_tracker import EMATracker
from dit_mc.training.pf_buffer import prefetch_single
from dit_mc.training.utils import get_optimizer, rotation_augmentation
from dit_mc.training.utils import make_update_fn
from dit_mc.training.checkpoint import create_checkpoint_manager_from_workdir, create_state_checkpoint_manager


class CustomTrainState(train_state.TrainState):
    rng: Any
    ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    epoch: int | jax.Array


class Trainer:
    def __init__(
            self, 
            process: GenerativeProcess, 
            data_module,
            seed, 
            num_epochs, 
            workdir, 
            augmentation_bool,
            use_validation_bool,
            log_every_t=20,
            save_every_t=5,
            validate_every_t=5,
            reset_to_ema_every_epoch=False, 
            resume_from_checkpoint=False,
    ):
        self.process = process
        self.data_module = data_module
        self.num_epochs = num_epochs
        self.num_steps_total_approx = ((self.data_module.get_len("train") * self.num_epochs) // (max(2, self.data_module.max_num_graphs) - 1)) + 1
        self.num_steps_per_epoch = ((self.data_module.get_len("train")) // (max(2, self.data_module.max_num_graphs) - 1)) + 1
        self.opt, self.lr_schedule = get_optimizer(num_steps_total=self.num_steps_total_approx)
        self.rng = random.PRNGKey(seed)
        self.seed = seed

        self.augmentation_bool = augmentation_bool
        self.use_validation_bool = use_validation_bool

        self.global_norm_fn = jax.jit(optax.global_norm)
        self.loss_fn = jax.jit(process.get_loss_fn())
        self.loss_and_grad_fn = jax.jit(jax.value_and_grad(process.get_loss_fn()))
        self.update_fn = jax.jit(
            make_update_fn(
                optimizer=self.opt,
                ema_bool=False,
            )
        )
        self.rot_aug = jax.jit(rotation_augmentation)
        self.reset_to_ema_every_epoch = reset_to_ema_every_epoch

        self.ckpt_mngr = create_checkpoint_manager_from_workdir(
            workdir=workdir,
            create_ckpt_dir=not resume_from_checkpoint,
            ckpt_mngr_options=dict(
                max_to_keep=num_epochs//save_every_t,
                save_interval_steps=save_every_t
            )
        )

        self.ckpt_mngr_last = create_checkpoint_manager_from_workdir(
            workdir=workdir,
            create_ckpt_dir=not resume_from_checkpoint,
            ckpt_dir_name='last_checkpoint',
            ckpt_mngr_options=dict(
                max_to_keep=1,
                save_interval_steps=1
            )
        )

        self.ckpt_mngr_no_ema = create_checkpoint_manager_from_workdir(
            workdir=workdir,
            create_ckpt_dir=not resume_from_checkpoint,
            ckpt_dir_name='checkpoints_no_ema',
            ckpt_mngr_options=dict(
                max_to_keep=num_epochs//save_every_t,
                save_interval_steps=save_every_t
            )
        )

        self.ckpt_mngr_val_no_ema = create_checkpoint_manager_from_workdir(
            workdir=workdir,
            create_ckpt_dir=not resume_from_checkpoint,
            ckpt_dir_name='best_val_loss_checkpoint_no_ema',
            ckpt_mngr_options=dict(
                max_to_keep=1,
                save_interval_steps=1
            )
        )

        self.ckpt_mngr_last_no_ema = create_checkpoint_manager_from_workdir(
            workdir=workdir,
            create_ckpt_dir=not resume_from_checkpoint,
            ckpt_dir_name='last_checkpoint_no_ema',
            ckpt_mngr_options=dict(
                max_to_keep=1,
                save_interval_steps=1
            )
        )

        self.ckpt_mngr_restore = create_state_checkpoint_manager(
            workdir=workdir,
            create_ckpt_dir=not resume_from_checkpoint,
            ckpt_dir_name='restore_checkpoint',
            ckpt_mngr_options=dict(
                max_to_keep=1,
                save_interval_steps=1
            )
        )

        self.ckpt_mngrs = [
            self.ckpt_mngr,
            self.ckpt_mngr_last,
            self.ckpt_mngr_no_ema,
            self.ckpt_mngr_val_no_ema,
            self.ckpt_mngr_last_no_ema,
            self.ckpt_mngr_restore,
        ]

        self.log_every_t = log_every_t
        self.save_every_t = save_every_t
        self.validate_every_t = validate_every_t
        self.resume_from_checkpoint = resume_from_checkpoint

    def init_params(self):
        self.rng, rng_init = jax.random.split(self.rng)
        graph, graph_cond, _ = self.data_module.get_sample()
        graph.nodes['positions'] = np.zeros_like(graph.nodes['x1'])
        graph.nodes['cond_scaling'] = np.zeros(len(graph.nodes['x1'],))
        graph.edges['cond_scaling'] = np.zeros(len(graph.senders,))
        if self.process.self_conditioning_bool:
            graph.nodes['self_cond'] = np.zeros_like(graph.nodes['x1']) # TODO: (num_nodes, 6) for TSI
        params = self.process.net.init(
            rng_init,
            graph_latent=graph,
            graph_cond=graph_cond if self.process.conditioning_bool else None,
            time_latent=np.ones(len(graph.nodes['x1'],)),
        )
        del graph.nodes['positions']
        del graph.nodes['cond_scaling']
        del graph.edges['cond_scaling']
        if self.process.self_conditioning_bool:
            del graph.nodes['self_cond']
        return params
    
    def finish(self):
        for c in self.ckpt_mngrs:
            c.wait_until_finished()
            c.close()

    def validate(self, params, epoch, step):
        print("Validating...")
        losses = []
        val_rng = random.PRNGKey(self.seed)

        batch = self.data_module.next_epoch("val")
        batch = prefetch_single(batch, size=8) # prefetch to device

        for graph, graph_cond, graph_prior in tqdm(batch):
            rng_loss, val_rng = jax.random.split(val_rng, num=2)
            loss = self.loss_fn(
                params,
                rng=rng_loss,
                graph=graph,
                graph_prior=graph_prior,
                graph_cond=graph_cond,
            )

            losses.append(loss)

        loss = np.stack(losses).mean()

        wandb.log({
            "val/epoch": epoch,
            "val/global_step": step,
            "val/loss": loss,
        })

        return loss
    
    def get_train_state(self, params, ema_params, opt_state, step, epoch):
        return CustomTrainState(
            params=jax.tree.map(lambda x: x, params),
            opt_state=jax.tree.map(lambda x: x, opt_state),
            ema_params=jax.tree.map(lambda x: x, ema_params),
            rng=self.rng,
            step=step,
            epoch=epoch,
            apply_fn=None,
            tx=self.opt
        )
    
    def save_train_state(self, params, ema_params, opt_state, step, epoch):
        state = self.get_train_state(
            params=params,
            ema_params=ema_params,
            opt_state=opt_state,
            step=step,
            epoch=epoch
        )
        self.ckpt_mngr_restore.save(epoch, {"state": state})

    def restore_train_state(self, params, opt_state):
        dummy_state = self.get_train_state(
            params=params,
            ema_params=params,
            opt_state=opt_state,
            step=0,
            epoch=0
        )

        return self.ckpt_mngr_restore.restore(
            self.ckpt_mngr_restore.latest_step(),
            items={"state": dummy_state}
        )["state"]

    def train(self, params):
        step = 0
        start_epoch = 0
        best_loss = 1e6
        best_params = None
        best_val_loss = 1e6
        best_val_params = None
        opt_state = self.opt.init(params)
        ema = EMATracker()
        ema.initialize(params)

        if self.resume_from_checkpoint:
            print("Restoring from checkpoint...")
            state = self.restore_train_state(params, opt_state)

            params = state.params
            opt_state = state.opt_state
            self.rng = state.rng
            step = state.step
            start_epoch = state.epoch
            ema.initialize(state.ema_params)
        else:
            # save the first resume checkpoint after initialization
            self.save_train_state(params, ema.shadow_params, opt_state, step, 0)

        for epoch in range(start_epoch, self.num_epochs):
            print(f"Starting Epoch {epoch}...")
            batch = self.data_module.next_epoch("train")
            batch = prefetch_single(batch, size=8) # prefetch to device

            for graph, graph_cond, graph_prior in (pbar := tqdm(batch, total=self.num_steps_per_epoch)):
                # for logging purpose only
                lr = self.lr_schedule(step)
                rng_loss, self.rng = jax.random.split(self.rng, num=2)
               
                # Whe only consider the bonding graph, so it has no positions.
                # But it might be important later to align graph and graph_cond
                # if self.augmentation_bool:
                #     rng_rot, self.rng = jax.random.split(self.rng, num=2)
                #     graph, rot = rotation_augmentation(rng=rng_rot, graph=graph, return_rotation=True)
                #     graph_cond = rotate_graph(graph=graph_cond, rotation=rot)

                if self.augmentation_bool:
                    rng_rot, self.rng = jax.random.split(self.rng, num=2)
                    graph = self.rot_aug(rng=rng_rot, graph=graph)

                loss, grads = self.loss_and_grad_fn(
                    params,
                    rng=rng_loss,
                    graph=graph,
                    graph_prior=graph_prior,
                    graph_cond=graph_cond,
                )

                params, opt_state = self.update_fn(
                    params=params,
                    grads=grads,
                    optimizer_state=opt_state
                )
                ema.update(params)

                if loss < best_loss:
                    best_params = params.copy()
                    best_loss = loss

                if step % self.log_every_t == 0:
                    wandb.log({
                        "train/epoch": epoch,
                        "train/global_step": step,
                        "train/loss_step": loss,
                        "train/learning_rate": lr,
                        "train/gradient_norm": self.global_norm_fn(grads),
                    })
                    pbar.set_postfix({"loss": loss})

                step += 1

            # at the end of each epoch, reset params to EMA
            if self.reset_to_ema_every_epoch:
                params = jax.tree.map(lambda x: x, ema.shadow_params)

            # at the end of epoch, run validation for EMA params
            if self.use_validation_bool and epoch % self.validate_every_t == 0:
                val_params = jax.tree.map(lambda x: x, params)
                val_loss = self.validate(val_params, epoch, step)
                if val_loss < best_val_loss:
                    best_val_params = params.copy()
                    best_val_loss = val_loss
                    self.ckpt_mngr_val_no_ema.save(
                        epoch,
                        args=checkpoint.args.Composite(params=checkpoint.args.StandardSave(jax.tree.map(lambda x: x, params))),
                    )
            
            if (epoch + 1) % self.save_every_t == 0:
                # save params without EMA once every t epochs (keep all ckpts)
                self.ckpt_mngr_no_ema.save(
                    epoch+1,
                    args=checkpoint.args.Composite(params=checkpoint.args.StandardSave(jax.tree.map(lambda x: x, params))),
                )

                # save EMA params once every t epochs (keep all ckpts)
                self.ckpt_mngr.save(
                    epoch+1,
                    args=checkpoint.args.Composite(params=checkpoint.args.StandardSave(jax.tree.map(lambda x: x, ema.shadow_params))),
                )

            # save params without EMA once per epoch (keep last ckpt only)
            self.ckpt_mngr_last_no_ema.save(
                epoch+1,
                args=checkpoint.args.Composite(params=checkpoint.args.StandardSave(jax.tree.map(lambda x: x, params))),
            )

            # save EMA params once per epoch (keep last ckpt only)
            self.ckpt_mngr_last.save(
                epoch+1,
                args=checkpoint.args.Composite(params=checkpoint.args.StandardSave(jax.tree.map(lambda x: x, ema.shadow_params))),
            )

            self.save_train_state(params, ema.shadow_params, opt_state, step, epoch+1)

        self.finish()
        return best_params, best_val_params
