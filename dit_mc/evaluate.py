import re
import os
import glob
import wandb
import argparse
from tqdm.auto import tqdm
from functools import partial
from multiprocessing import Pool
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D

from dit_mc.data_loader.utils import load_pkl, save_pkl


def mol_from_xyz(atomic_numbers, coords):
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atomic_numbers))

    for i, (z, pos) in enumerate(zip(atomic_numbers, coords)):
        atom = Chem.Atom(int(z))
        mol_idx = mol.AddAtom(atom)
        conf.SetAtomPosition(mol_idx, Point3D(*pos))

    mol = mol.GetMol()
    mol.AddConformer(conf)
    return mol


def get_best_rmsd(ref_mol, gen_mol, use_alignmol=False):
    try:
        if use_alignmol:
            return rdMolAlign.AlignMol(gen_mol, ref_mol)
        else:
            rmsd = rdMolAlign.GetBestRMS(gen_mol, ref_mol)
    except:  # noqa
        rmsd = np.nan

    return rmsd


def worker_fn(job, use_alignmol=False):
    smiles, ref_idx, gen_idx, ref_mol, gen_mol = job
    rmsd_val = get_best_rmsd(ref_mol, gen_mol, use_alignmol=use_alignmol)
    return smiles, ref_idx, gen_idx, rmsd_val


def calc_coverage_recall(rmsd_array, thresholds):
    min_rmsd_per_conf = np.nanmin(rmsd_array, axis=1, keepdims=True) # (num_confs, 1)
    hits_per_conf = min_rmsd_per_conf < thresholds # (num_confs, num_thresholds)
    coverage_recall = np.mean(hits_per_conf, axis=0) # (num_thresholds,)
    return coverage_recall


def calc_coverage_precision(rmsd_array, thresholds):
    thresholds = np.expand_dims(thresholds, 1) # (num_thresholds, 1)
    min_rmsd_per_pred = np.nanmin(rmsd_array, axis=0, keepdims=True) # (1, num_preds)
    hits_per_pred = min_rmsd_per_pred < thresholds # (num_thresholds, num_preds)
    coverage_precision = np.mean(hits_per_pred, axis=1) # (num_thresholds,)
    return coverage_precision


def calc_amr_recall(rmsd_array):
    min_rmsd_per_conf = np.nanmin(rmsd_array, axis=1) # (num_confs,)
    amr_recall = np.mean(min_rmsd_per_conf) # ()
    return amr_recall


def calc_amr_precision(rmsd_array):
    min_rmsd_per_pred = np.nanmin(rmsd_array, axis=0) # (num_preds,)
    amr_precision = np.mean(min_rmsd_per_pred) # ()
    return amr_precision


def print_covmat_results(results, step, threshold):

    df = pd.DataFrame.from_dict(
        {
            "Threshold": results["thresholds"],
            "COV-R_mean": np.mean(results["CoverageR"], 0),
            "COV-R_median": np.median(results["CoverageR"], 0),
            "COV-P_mean": np.mean(results["CoverageP"], 0),
            "COV-P_median": np.median(results["CoverageP"], 0),
        }
    )

    mask = np.abs(results['thresholds'] - threshold) < 1e-6

    metrics = {
        "test/epoch": np.array([step]),
        "test/COV-R_mean": df["COV-R_mean"][mask].to_numpy(),
        "test/COV-R_median": df["COV-R_median"][mask].to_numpy(),
        "test/COV-P_mean": df["COV-P_mean"][mask].to_numpy(),
        "test/COV-P_median": df["COV-P_median"][mask].to_numpy(),
        "test/MAT-R_mean": np.mean(results["MatchingR"]),
        "test/MAT-R_median": np.median(results["MatchingR"]),
        "test/MAT-P_mean": np.mean(results["MatchingP"]),
        "test/MAT-P_median": np.median(results["MatchingP"]),
    }

    return df, metrics


class CovMatEvaluator(object):
    """Coverage Recall Metrics Calculation for GEOM-Dataset"""

    def __init__(
        self,
        num_workers: int = 8,
        use_alignmol: bool = False,
        allow_permutations: bool = False,
        thresholds: np.ndarray = np.arange(0.05, 3.05, 0.05),
    ):
        super().__init__()
        self.num_workers = num_workers
        self.use_alignmol = use_alignmol
        self.allow_permutations = allow_permutations
        self.thresholds = np.array(thresholds).flatten()

    def __call__(self, mols_true, mols_pred):
        jobs = []
        rmsd_results = {}
        for smiles in list(mols_pred.keys()):

            num_true = len(mols_true[smiles])
            num_pred = len(mols_pred[smiles])
            rmsd_results[smiles] = np.nan * np.ones((num_true, num_pred))

            ref_mols = []
            for ref_mol in mols_true[smiles]:
                ref_mol = Chem.rdmolops.RemoveHs(ref_mol)
                if not self.use_alignmol and self.allow_permutations:
                    atomic_numbers = [atom.GetAtomicNum() for atom in ref_mol.GetAtoms()]
                    coords = ref_mol.GetConformer().GetPositions()
                    ref_mol = mol_from_xyz(atomic_numbers, coords)
                ref_mols.append(ref_mol)

            gen_mols = []
            for gen_mol in mols_pred[smiles]:
                gen_mol = Chem.rdmolops.RemoveHs(gen_mol)
                if not self.use_alignmol and self.allow_permutations:
                    atomic_numbers = [atom.GetAtomicNum() for atom in gen_mol.GetAtoms()]
                    coords = gen_mol.GetConformer().GetPositions()
                    gen_mol = mol_from_xyz(atomic_numbers, coords)
                gen_mols.append(gen_mol)

            for ref_idx, ref_mol in enumerate(ref_mols):
                for gen_idx, gen_mol in enumerate(gen_mols):
                    jobs.append((smiles, ref_idx, gen_idx, ref_mol, gen_mol))

        def populate_results(res):
            smiles, ref_idx, gen_idx, rmsd_val = res
            rmsd_results[smiles][ref_idx, gen_idx] = rmsd_val

        if self.num_workers > 1:
            p = Pool(self.num_workers)
            map_fn = partial(p.imap_unordered, chunksize=64)
            p.__enter__()
        else:
            map_fn = map

        fn = partial(worker_fn, use_alignmol=self.use_alignmol)

        for res in tqdm(map_fn(fn, jobs), total=len(jobs)):
            populate_results(res)

        if self.num_workers > 1:
            p.__exit__(None, None, None)

        amr_recall, amr_precision = [], []
        coverage_recall, coverage_precision = [], []
        for rmsd_array in rmsd_results.values():
            amr_recall.append(calc_amr_recall(rmsd_array))
            amr_precision.append(calc_amr_precision(rmsd_array))
            coverage_recall.append(
                calc_coverage_recall(rmsd_array, self.thresholds))
            coverage_precision.append(
                calc_coverage_precision(rmsd_array, self.thresholds))

        results = {
            "thresholds": self.thresholds,
            "CoverageR": np.array(coverage_recall),
            "MatchingR": np.array(amr_recall),
            "CoverageP": np.array(coverage_precision),
            "MatchingP": np.array(amr_precision),
        }

        return results, rmsd_results


def evaluate_main(workdir, step, seed, num_workers=1, use_alignmol=False, allow_permutations=False, use_correction=True):
    cfg = OmegaConf.load(os.path.join(workdir, ".hydra", "config.yaml"))

    matches = glob.glob(os.path.join(workdir, "wandb", "latest-run", "run-*.wandb"))
    if matches:
        filename = os.path.basename(matches[0])
        wandb_run_id = re.findall(r'run-([a-z0-9]+)\.wandb', filename)[0]
    else:
        raise Exception("Can't retrieve wandb_run_id. No matching wandb log found.")

    if wandb.run is not None:
        wandb.finish()

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=wandb_run_id,
        resume="must",
        dir=workdir,
    )

    if step == -1:
        pattern = os.path.join(workdir, "pred_mols_epoch*.pkl")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No files found matching pattern {pattern}")
        steps = [int(re.search(r"epoch(\d+)", f).group(1)) for f in files]
        step = max(steps)
        print(f"Using latest step: {step}")
    
    used_cc = "CC" if use_correction else "plain"
    if seed == -1:
        file_key = f"{step}_{used_cc}"
    else:
        file_key = f"{step}_seed{seed}_{used_cc}"

    predicted_mols_path = os.path.join(workdir, f"pred_mols_epoch{file_key}.pkl")
    os.path.exists(predicted_mols_path), f"Path {predicted_mols_path} does not exist."

    datadir = cfg.data_loader.data_cfg["data_dir"]
    dataset = cfg.data_loader.data_cfg["dataset"]
    ground_truth_mols_path = os.path.join(datadir, 'geom', dataset, 'test_mols.pkl')
    os.path.exists(ground_truth_mols_path), f"Path {ground_truth_mols_path} does not exist."

    print("Load data...")
    predicted_mols = load_pkl(predicted_mols_path)
    ground_truth_mols = load_pkl(ground_truth_mols_path)

    print("Run evaluation...")
    evaluator = CovMatEvaluator(
        num_workers=num_workers, 
        use_alignmol=use_alignmol, 
        allow_permutations=allow_permutations
    )
    results, rmsd_results = evaluator(ground_truth_mols, predicted_mols)

    print("Save results...")
    threshold = 0.5 if dataset == "qm9" else 0.75
    cov_df, metrics = print_covmat_results(results, step, threshold)
    
    # Save results to a CSV file
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df.to_csv(os.path.join(workdir, f"metrics_epoch{file_key}.csv"), index=False) 
    cov_df.to_csv(os.path.join(workdir, f"cov_results_epoch{file_key}.csv"), index=False)
    # END CSV

    table = wandb.Table(dataframe=cov_df)
    wandb.run.log({f"Metrics (run_id: {wandb_run_id}, step: {step}, CC: {use_correction})": table})
    wandb.run.log(metrics)

    rmsd_values = []
    rmsd_values_per_gen = []
    rmsd_values_per_mol = []
    for rmsd in rmsd_results.values():
        rmsd_per_mol = np.min(rmsd, axis=1)
        rmsd_values_per_mol.extend(rmsd_per_mol[~np.isnan(rmsd_per_mol)].tolist())
        rmsd_per_gen = np.min(rmsd, axis=0)
        rmsd_values_per_gen.extend(rmsd_per_gen[~np.isnan(rmsd_per_gen)].tolist())
        rmsd = rmsd[~np.isnan(rmsd)]
        rmsd_values.extend(rmsd.tolist())

    save_pkl(os.path.join(workdir, f"rmsd_values_epoch{file_key}.pkl"), rmsd_values)
    save_pkl(os.path.join(workdir, f"rmsd_values_per_mol_epoch{file_key}.pkl"), rmsd_values_per_mol)
    save_pkl(os.path.join(workdir, f"rmsd_values_per_gen_epoch{file_key}.pkl"), rmsd_values_per_gen)

    fig, ax = plt.subplots()
    ax.hist(rmsd_values, bins=50, alpha=0.5)
    ax.set_xlabel("RMSD")
    ax.set_ylabel("Frequency")
    ax.set_title(f"RMSD Histogram")
    wandb.run.log({f"RMSD Histogram (run_id: {wandb_run_id}, step: {step}, CC: {use_correction})": wandb.Image(fig)})
    
    # Save the histogram figure to the workdir
    fig_path = os.path.join(workdir, f"rmsd_histogram_epoch{file_key}.png")
    fig.savefig(fig_path)
    plt.close(fig)

    wandb.finish()
    print("Done.")


if __name__ == "__main__":    
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", 
        type=str, 
        help="Path of the directory containing the model parameters.", 
        required=True
    )
    parser.add_argument(
        "--step", 
        type=int, 
        default=-1, 
        help="Checkpoint step.",
        required=False
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=-1, 
        help="Seed. Default -1 (backward compatibility).",
        required=False
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=-1, 
        help="Number of workers. Default 1."
    )
    parser.add_argument(
        "--use_alignmol",
        action="store_true",
        default=False,
        help="Use alignmol for matching",
    )
    parser.add_argument(
        "--allow_permutations",
        action="store_true",
        default=False,
        help="Use permutation-aware RSMD metric.",
    )
    parser.add_argument(
        "--use_correction",
        action="store_true",
        default=True,
        help="Load checkpoint with chirality correction.",
    )
    args = parser.parse_args()

    workdir = args.workdir
    step = args.step
    seed = args.seed
    num_workers = args.num_workers
    use_alignmol = args.use_alignmol
    allow_permutations = args.allow_permutations
    use_correction = args.use_correction

    if num_workers == -1:
        num_proc = len(os.sched_getaffinity(0))
        num_workers = min(50, num_proc)
        print(f"## Found {num_proc} cpus and using {num_workers} workers. ##")
    
    assert num_workers <= 50, "Too many workers cause issues with open file handles."
    evaluate_main(
        workdir=workdir, 
        step=step, 
        seed=seed,
        num_workers=num_workers, 
        use_alignmol=use_alignmol, 
        allow_permutations=allow_permutations,
        use_correction=use_correction
    )
