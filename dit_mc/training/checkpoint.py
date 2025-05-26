from orbax import checkpoint
from pathlib import Path
from typing import Dict
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointHandler, CheckpointManagerOptions


def create_checkpoint_manager_from_workdir(
        workdir,
        ckpt_dir_name: str = 'checkpoints',
        create_ckpt_dir: bool = True,
        ckpt_mngr_options: Dict = None
):

    ckpt_dir = Path(workdir) / ckpt_dir_name

    if create_ckpt_dir is True:
        ckpt_dir = Path(ckpt_dir).expanduser().resolve()
        ckpt_dir.mkdir(exist_ok=False)

    if ckpt_mngr_options is None:
        ckpt_mngr_options = {
            'max_to_keep': 1,
            'save_interval_steps': 10_000
        }
    else:
        pass

    options = checkpoint.CheckpointManagerOptions(
        step_prefix='ckpt',
        **ckpt_mngr_options
    )

    ckpt_mngr = checkpoint.CheckpointManager(
        ckpt_dir,
        item_names=('params',),
        item_handlers={'params': checkpoint.StandardCheckpointHandler()},
        options=options
    )

    return ckpt_mngr


def create_state_checkpoint_manager(        
        workdir,
        ckpt_dir_name: str = 'restore_checkpoint',
        create_ckpt_dir: bool = True,
        ckpt_mngr_options: Dict = None
):
    ckpt_dir = Path(workdir) / ckpt_dir_name

    if create_ckpt_dir is True:
        ckpt_dir = Path(ckpt_dir).expanduser().resolve()
        ckpt_dir.mkdir(exist_ok=False)

    options = checkpoint.CheckpointManagerOptions(
        step_prefix='ckpt',
        **ckpt_mngr_options
    )

    ckpt_mngr = checkpoint.CheckpointManager(
        ckpt_dir,
        item_names=('state',),
        item_handlers={'state': checkpoint.StandardCheckpointHandler()},
        options=options
    )

    return ckpt_mngr


def load_params_from_workdir(
        workdir,
        ckpt_dir_name: str = 'checkpoints',
        step: int = -1,
):
    loaded_mngr = create_checkpoint_manager_from_workdir(
        workdir,
        ckpt_dir_name=ckpt_dir_name,
        create_ckpt_dir=False
    )

    if step == -1:
        step = loaded_mngr.latest_step()

    mngr_state = loaded_mngr.restore(
        step,
    )

    params = mngr_state.get('params')

    return params, step
