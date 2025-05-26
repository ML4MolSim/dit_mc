import hydra
from omegaconf import DictConfig, OmegaConf


def get_hydra_output_dir():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # first set the default context to spawn on the compute node
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    import socket
    import sys

    hostname = socket.gethostname()
    print(f"Running on machine: {hostname}")

    if hostname == "hydra":
        print("Trying to run experiment on login node. Please use --multirun option to submit jobs.")
        exit()

    import wandb
    import jax
    from absl import logging
    import os
    import glob
    import re

    logging.set_verbosity(logging.WARNING)

    resume_training = False
    workdir = get_hydra_output_dir()
    if cfg.resume_from_workdir:
        # Resume from a previous run
        print("Resume from previous run...")
        workdir = cfg.resume_from_workdir
        assert workdir is not None, "Please provide a workdir to resume from."

        cfg = OmegaConf.load(os.path.join(workdir, ".hydra", "config.yaml"))
        
        matches = glob.glob(os.path.join(workdir, "wandb", "latest-run", "run-*.wandb"))
        if matches:
            filename = os.path.basename(matches[0])
            wandb_run_id = re.findall(r'run-([a-z0-9]+)\.wandb', filename)[0]
        else:
            raise Exception("Can't retrieve wandb_run_id. No matching wandb log found.")

        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=wandb_run_id,
            resume="must",
            dir=workdir,
        )
        resume_training = True
    else:
        # Start a fresh run
        print("Instantiate logging...")
        config = OmegaConf.to_container(cfg, resolve=True)
        config["workdir"] = workdir
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=f"EXP-{cfg.globals.exp}",
            dir=workdir,
            config=config,
            mode=cfg.wandb.mode
        )

    print("Instantiate data module...")
    data_module = hydra.utils.instantiate(cfg.data_loader)

    if cfg.globals.download_data:
        data_module.download()
    
    print("Instantiate trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        data_module=data_module,
        seed=cfg.globals.seed, 
        workdir=workdir,
        resume_from_checkpoint=resume_training,
    )

    try:
        print("Initialize parameters...")
        params = trainer.init_params()
        num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
        print(f"Number of parameters: {num_params / 1e6:.2f}M params")

        print("Start training...")
        trainer.train(params)
        print("Done.")
    except KeyboardInterrupt:
        print("\n[!] Caught CTRL+C — terminating...")
        data_module.shutdown()
        sys.exit(1)
    except Exception as e:
        print(f"[!] Error: {e}")
        data_module.shutdown()
        raise e

    data_module.shutdown()
    wandb.finish()

    # auto eval
    if cfg.globals.auto_eval:
        print("Starting evaluation...")
        from dit_mc.generate_confs import generate_confs_main
        from dit_mc.evaluate import evaluate_main

        step = generate_confs_main(workdir=workdir, ckpt_dir_name="last_checkpoint")
        evaluate_main(workdir=workdir, step=step, num_workers=cfg.globals.n_cpus)

if __name__ == "__main__":
    main()
