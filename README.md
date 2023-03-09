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

DiscoTope-3.0 is a structure-based B-cell epitope prediction tool, exploiting inverse folding latent representations from the [ESM-IF1](https://github.com/facebookresearch/esm) model. The tool accepts input protein structures in the [PDB](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) format (solved or predicted), and outputs per-residue epitope propensity scores in both a PDB and CSV format.

DiscoTope-3.0 accepts both experimental and AlphaFold2 modeled structures, with similar performance for both. It has been trained and validated only on single chain structures, meaning epitopes may be predicted in interface regions.

- Paper: [10.1101/2023.02.05.527174](https://www.biorxiv.org/content/10.1101/2023.02.05.527174v1)
- Full training/validation/testing datasets: https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0
- Web server DTU: https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0
- Web server BioLib: https://biolib.com/DTU/DiscoTope-3/

# Repo contents

- [data](./data): Test set and example antigen PDB files
- [src](./src): Source code
- [output](./output): DiscoTope-3.0 output examples
- [help.py](./help.py): (Optional) interactive helper script for running DiscoTope-3.0

# System Requirements

## Hardware Requirements

For minimal performance, only a single core and ca 8 GB of RAM is needed. For optional performance, we recommend the following specs:

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
- [fastpdb](https://github.com/biotite-dev/fastpdb)
- [prody](https://github.com/prody/ProDy)
- [pandas](https://github.com/pandas-dev/pandas)
- [numpy](https://github.com/numpy/numpy)

# Installation guide

## Installing on Ubuntu 18.04 (~ 2 mins)

```bash
# Setup environment with conda
conda create -n inverse python=3.9
conda activate inverse
conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch   ## very important to specify pytorch package!
conda install pyg -c pyg -c conda-forge ## very important to make sure pytorch and cuda versions not being changed
conda install pip

# further requirements
pip install -r requirements.txt

# Unzip model weights
unzip models.zip
```

For GPU accelerated predictions, please install [py-xgboost-gpu](https://xgboost.readthedocs.io/en/stable/install.html) and make sure a GPU is available.

```bash
conda install -c conda-forge py-xgboost-gpu
```

# Demo

On a common workstation with a GPU, predictions takes ~ 1 second per PDB chain with ~ 15 seconds for loading needed libraries and model weight. Ensure model weights are unzipped by first running 'unzip models.zip' (see [Installation Guide](#installation-guide))

## Predict a single PDB (solved structure)

```bash
python src/predict_webserver.py \
--pdb_or_zip_file data/example_pdbs_solved/7lkh.pdb \
--struc_type solved \
--out_dir output/7lkh
```

## Reproduce test-set predictions (AlphaFold structures)

```bash
python src/predict_webserver.py \
--pdb_dir data/test_set_solved \
--struc_type alphafold \
--out_dir output/test_set_solved
```

## Running on own data

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

DiscoTope-3.0 outputs per-residue epitope propensity scores inside single chain PDB files, with matching per-residue CSV files.

The CSV output files contains per-residue outputs, with the following column headers:
- PDB ID and chain name
- Relative residue index (re-numbered from 1)
- Amino-acid residue, 1-letter
- DiscoTope-3.0 score
- Relative surface accessibility (Shrake-Rupley, normalized using Sander scale)
- AlphaFold pLDDT score (set to 100 for non-AlphaFold structures)
- Chain length
- A binary feature set to 1 for AlphaFold structures.

Example output CSV (see [output/7lkh_discotope3.csv](./output/7lkh_discotope3.csv)):
```text
pdb,res_id,residue,DiscoTope-3.0_score,rsa,pLDDTs,length,alphafold_struc_flag
7lkh,53,G,0.04832,1.30833,47.17,516,1
7lkh,54,P,0.04014,0.26530,42.35,516,1
7lkh,55,V,0.04250,0.43499,41.18,516,1
7lkh,56,E,0.10191,0.77296,44.61,516,1
```

The PDB output files contain the original PDB information, with the B-factor column replaced with per-residue DiscoTope-3.0 scores (2nd right-most column).

Example output PDB (see [output/7lkh_A.pdb](./output/7lkh_A_discotope3.pdb)):
```text
REMARK AtomGroup 7lkh_A
ATOM      1  N   PRO A   1     -16.036  -6.927  16.692  1.00 22.10           N  
ATOM      2  CA  PRO A   1     -15.169  -8.075  16.418  1.00 22.10           C  
ATOM      3  C   PRO A   1     -14.773  -8.177  14.946  1.00 22.10           C  
```

# Common issues

- Breaks on single chains with invalid amino-acid residues in the extracted backbone (solved structures only)
- PDBConstructionWarning regarding discontinuous chains: Indicates missing residue atoms in the input PDB file. Common issue for solved structures. May impact impact DiscoTope-3.0 performance (solved structures only)
- Biopython and ESM future deprecation warnings: Benign library warnings, does not impact predictions
- ESM regression weights missing warning: Benign fair-esm library warning, does not impact predicitons

# Citation
For usage of the package and associated manuscript, please cite according to the enclosed [citation.bib](./citation.bib).

# License

This source code is licensed under the Creative Commons license found in the [LICENSE](./LICENSE) file in the root directory of this source tree.
