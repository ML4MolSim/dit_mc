[![data](https://zenodo.org/badge/DOI/10.5281/zenodo.14779793.svg)](https://doi.org/10.5281/zenodo.15489212)

# DiTMC - Sampling 3D Molecular Conformers with Diffusion Transformers

<img src="https://github.com/ML4MolSim/dit_mc/blob/main/ditmc_ani.gif" width="500" height="200">

DiTMC is a modular framework for molecular conformer generation using Diffusion Transformers (DiTs). 
It integrates molecular graph information with 3D geometry through flexible graph-based conditioning and supports both non-equivariant and SO(3)-equivariant attention. 
DiTMC achieves state-of-the-art results on the GEOM benchmarks.

## Overview
We use modular hydra configs to ablate parts of our architecture or algorithms. Config files can be found in `configs` folder.

Our modular repository consists of 3 main entry points:
- `dit_mc/app.py`: the main entry point for training. 
- `dit_mc/generate_confs.py`: script for sampling conformers from pre-trained checkpoints.
- `dit_mc/evaluate.py`: script for evaluating conformers against reference data.

Our repository is further organized in the following way:
- `dit_mc/backbones`: our architectural blocks that can be used to assemble the DiTMC models.
- `dit_mc/data_loader`: a custom data loader for tfrecords, that supports custom batching using jraph tuples and multiprocessing.
- `dit_mc/generative_process`: our modular package structure contains an abstraction layer different generative algorithms, we support flow matching with constant noise for now.
- `dit_mc/model_zoo`: our assembled DitMC architectures.
- `dit_mc/training`: main components for training and validation, as well as EMA calculation and checkpointing.  
- `tf_datasets`: tensorflow dataset builder to run preprocessing and store GEOM QM9, Drugs and XL datasets as tfrecords.

## Preparation
For running the code, we recommend to download the code, data and checkpoints directly from our [anonymized zenodo link](https://doi.org/10.5281/zenodo.15489212).
This ensures that the data is located in `./data` and checkpoints are located in `./checkpoints` folders relative to the dit_mc repository.

## Install requirements
You can install our package directly via pip from source.

``` bash
pip install .
```

## Data preperation
Our code expects preprocessed data in tfrecord format for the GEOM datasets.
For the final version of the paper, we will provide preprocessed datasets for convenience in tfrecord format. 
For now, we provide tfrecords for testing via our [anonymized zenodo link](https://doi.org/10.5281/zenodo.15489212).
To run preprocessing yourself do the following:

Step 1 - Download and extract the GEOM dataset from the original source.
``` bash
wget https://dataverse.harvard.edu/api/access/datafile/4327252 -O ./data/geom/rdkit_folder.tar.gz
tar -xvf ./data/geom/rdkit_folder.tar.gz
```

Step 2 – Download the data splits from GeoMol and move them to the respective subfolders within the `./data/geom` directory.
```
https://github.com/PattanaikL/GeoMol/blob/main/data/QM9/splits/split0.npy
https://github.com/PattanaikL/GeoMol/blob/main/data/DRUGS/splits/split0.npy
```

Step 3 - Install requirements.
Besides the requirements above, we need additional requirements for generating the data:
``` bash
pip install mlcroissant apache-beam
```

Step 4 - Run the provided preprocessing script.
``` bash
cd ./tf_datasets/geom
bash build_ds.sh
```

Step 5 - Copy the generated tfrecord files to `./data`.

## Evaluation

We perform evaluation in a two-step procedure.

First, we generate conformers and save them as `$workdir/pred_mols.pkl`.
Second, we evaluate the saved predictions and save the results (metrics).
You should copy the provided tfrecord files (under `data/geom..`) for testing to `./data/geom..` in the folder of the repository for this to work.
You can specify one of the pre-trained checkpoints we provide for inference, e.g. `workdir={your-download-folder}/checkpoints/drugs/apeB`.

Example usage:
``` python
python generate_confs.py --workdir $workdir
python evaluate.py --workdir $workdir
```

## Training

During training, we use multiprocessing for distributed data loading using tensorflow datasets.
You can adjust the number of CPUs in hydra as globals.n_cpus.
The dataloader will start `n_cpus` parallel processes.

### Start Runs

We are ready to start runs now.

- You need to adapt the wandb entity in `config/config.yaml`. If you don't have a wandb account, you should create one.
- You can edit all settings in `configs/config.yaml` and start a run like this 
``` python
python dit_mc/app.py --multirun data_loader.data_cfg.data_dir=/tmp/data
```

## Citation

If you use parts of our code please cite our paper.
