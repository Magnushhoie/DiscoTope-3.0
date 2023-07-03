## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Results](#results)
- [Common issues](https://github.com/ebridge2/lol/issues)
- [Citation](#citation)
- [License](./LICENSE)

# Overview

[DiscoTope-3.0](https://services.healthtech.dtu.dk/services/DiscoTope-3.0/) is a structure-based B-cell epitope prediction tool, exploiting inverse folding latent representations from the [ESM-IF1](https://github.com/facebookresearch/esm) model. The tool accepts input protein structures in the [PDB](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) format (solved or predicted), and outputs per-residue epitope propensity scores in both a PDB and CSV format.

DiscoTope-3.0 accepts both experimental and AlphaFold2 modeled structures, with similar performance for both. It has been trained and validated only on single chain structures.

- Paper: [10.1101/2023.02.05.527174](https://www.biorxiv.org/content/10.1101/2023.02.05.527174v1)
- Datasets: https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0
- Web server DTU: https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0
- Web server BioLib: https://biolib.com/DTU/DiscoTope-3/

# Repo contents

- [data](./data): Test set and example antigen PDB files
- [src](./src): Source code
- [output](./output): DiscoTope-3.0 output examples
- [help.py](./help.py): (Optional) interactive helper script for running DiscoTope-3.0

# System Requirements

## Hardware Requirements

For minimal performance, only a single core and ca 8 GB of RAM is needed. For optimal performance, we recommend the following specs:

- RAM: 16+ GB
- CPU: 4+ cores
- GPU is optional

## Software Requirements

We highly recommend using an Ubuntu OS and Conda ([miniconda](https://docs.conda.io/en/main/miniconda.html) or [anaconda](https://www.anaconda.com/products/distribution)) for installing required dependencies. Exact versions of pytorch 1.11, cudatoolkit 11.3 and pytorch-geometric, scatter and sparse are required.

### OS Requirements

The package development version is tested on a *Linux* operating system. The developmental version of the package has been tested on the following systems:

Linux: Ubuntu 18.04

### Python requirements
- [Python 3.9](https://www.python.org/downloads/)
- [Pytorch 1.11](https://pytorch.org/get-started/locally/)
- [cudatoolkit 11.3](https://anaconda.org/anaconda/cudatoolkit)
- [Pytorch geometric 2.0.4](https://github.com/pyg-team/pytorch_geometric)
- [fair-esm 0.5](https://github.com/facebookresearch/esm)
- [Biopython](https://github.com/biopython/biopython)
- [Biotite](https://github.com/biotite-dev/biotite)
- [pandas](https://github.com/pandas-dev/pandas)
- [numpy](https://github.com/numpy/numpy)

# Installation guide

## Installing with conda (Linux) (recommended, ~2 mins) 

```bash
# Setup environment with conda
conda create -n inverse python=3.9
conda activate inverse
conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch   ## very important to specify pytorch package!
conda install pyg -c pyg -c conda-forge ## very important to make sure pytorch and cuda versions not being changed
conda install pip

# install pip dependencies
pip install -r requirements.txt

# Unzip models
unzip models.zip
```

## Installing with pip only (Linux) (~5 mins)
```bash
# install pip dependencies
pip install -r requirements_full.txt

# Unzip models
unzip models.zip
```

## Nb: gcc or g++ errors, missing torch-scatter build ...
```bash
# Make sure gcc and g++ versions are updated, pybind11 is available
# torch-scatter should be listed with 'conda list' or 'pip list'

# With conda:
conda install -c conda-forge pybind11 gcc cxx-compiler

# With apt-get
sudo apt-get install gcc g++
pip install pybind11
```

For GPU accelerated predictions, please install [py-xgboost-gpu](https://xgboost.readthedocs.io/en/stable/install.html) and make sure a GPU is available.

```bash
conda install -c conda-forge py-xgboost-gpu
```

# Demo

On a common workstation with a GPU, predictions takes <1 second per PDB chain with ~ 15 seconds for loading needed libraries and model weight. Ensure XGBoost model weights are unzipped by first running 'unzip models.zip' (see [Installation Guide](#installation-guide)). ESM-IF1 weights will be automatically downloaded the first time the prediction script is run (~ 1 min)

## Predict a single PDB (solved structure)

```bash
# Run on single PDB on CPU only (by default checks for available GPU)
python src/predict_webserver.py \
--cpu_mode \
--pdb_or_zip_file data/example_pdbs_solved/7c4s.pdb \
--struc_type solved \
--out_dir output/7c4s
```

## Reproduce test-set predictions (AlphaFold2 structures)

```bash
# Unzip AlphaFold2 test set
unzip data/test_set_af2.zip -d data/

# Run predictions on PDB folder
python src/predict_webserver.py \
--pdb_dir data/test_set_af2 \
--struc_type alphafold \
--out_dir output/test_set_af2
```

## Running on own data (batch-mode)

Set the --struc_type parameter to 'solved' for experimentally solved structures or 'alphafold' for modelled structures.

Note that DiscoTope-3.0 splits PDB structures into single chains before prediction.

```bash
# Predict on example (solved) PDBs in data/example_pdbs_solved folder
python src/predict_webserver.py \
--pdb_dir data/example_pdbs_solved \
--struc_type solved \
--out_dir output/example_pdbs_solved

# Fetch & predict PDBs from list file from AlphaFoldDB
python src/predict_webserver.py \
--list_file data/af2_list_uniprot.txt \
--struc_type alphafold \
--out_dir output/af2_list_uniprot

# Fetch & predict PDBs from list file from RCSB
python src/predict_webserver.py \
--list_file data/solved_list_rcsb.txt \
--struc_type solved \
--out_dir output/solved_list_rcsb

# See more options
python automate.py
```

# Results

DiscoTope-3.0 splits input PDBs into single-chain PDB files, then predict per-residue epitope propensity scores.
Outputs are saved in both PDB and CSV format.

The CSV output files contains per-residue outputs, with the following column headers:
- PDB ID and chain name
- Relative residue index (re-numbered from 1)
- Amino-acid residue, 1-letter
- DiscoTope-3.0 score (theoretical range 0.00 - 1.00)
- Relative surface accessibility (Shrake-Rupley, normalized using Sander scale)
- AlphaFold pLDDT score (0-100, set to 100 for non-AlphaFold structures)
- Chain length
- A binary feature set to 1 for AlphaFold structures.

The PDB output files contain individual single chains with the B-factor column replaced with per-residue DiscoTope-3.0 scores (2nd right-most column). Note that the scores are multiplied by 100 as PDB files only allow 2 decimals of precision.

Example input PDB (see [7c4s.pdb](./data/example_pdbs_solved/7c4s.pdb)):
```bash
python src/predict_webserver.py \
--pdb_or_zip_file data/example_pdbs_solved/7c4s.pdb \
--struc_type solved \
--out_dir output/7c4s
```

Example output CSV (see [7c4s_A_discotope3.csv](./output/7c4s/output/7c4s_A_discotope3.csv)):
```text
pdb,res_id,residue,DiscoTope-3.0_score,rsa,pLDDTs,length,alphafold_struc_flag
7c4s_A,14,G,0.15186,0.80634,100,282,0
7c4s_A,15,Q,0.13953,0.45077,100,282,0
7c4s_A,16,E,0.23955,0.72919,100,282,0
```

Example output PDB (see [7c4s_A_discotope3.pdb](./output/7c4s/output/7c4s_A_discotope3.pdb)):
```text
ATOM      1  N   GLY A  14     -16.773 -32.069  23.105  1.00 15.19           N  
ATOM      2  CA  GLY A  14     -15.595 -32.029  23.955  1.00 15.19           C  
ATOM      3  C   GLY A  14     -14.287 -31.844  23.204  1.00 15.19           C  
ATOM      4  O   GLY A  14     -13.284 -32.465  23.555  1.00 15.19           O  
```

# Common issues

- No valid amino-acid backbone found: Occurs if only heteroatoms (non-amino acid residues) are found in the extracted chain. DiscoTope-3.0 requires full amino-acid backbone C, Ca and N atoms.
- PDBConstructionWarning regarding discontinuous chains: Indicates missing residue atoms in the input PDB file. May impact DiscoTope-3.0 performance (solved structures only)
- Biopython future deprecation warning: Benign Biopython library warning, does not impact predictions
- ESM regression weights missing warning: Benign fair-esm library warning, does not impact predictions

# Note on reproducibility
Output is deterministic, i.e. the same machine will always produce the same output. However, if comparing results run on an older CUDA version or GPU, minor discrepancies in DiscoTope-3.0 scores may occur from the 4th significant figure e.g. 0.27130 -> 0.27125. These differences are due to inherent variability in floating point computations, arising especially from changes in algorithms / optimizatons across CUDA toolkit versions. 

# Citation
For usage of the package and associated manuscript, please cite according to the enclosed [citation.bib](./citation.bib).

# License

This source code is licensed under the Creative Commons license found in the [LICENSE](./LICENSE) file in the root directory of this source tree.
