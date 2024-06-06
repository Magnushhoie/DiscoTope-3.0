# Overview

DiscoTope-3.0 predicts epitopes on input protein structures, using inverse folding representations from the [ESM-IF1](https://github.com/facebookresearch/esm) model.
The tool accepts both solved and predicted structures in the PDB format, and outputs per-residue epitope propensity scores in a CSV format.

- Paper: [10.3389/fimmu.2024.1322712](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2024.1322712/full)
- Datasets: [https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0](https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0)
- Web server DTU: [https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0](https://services.healthtech.dtu.dk/services/DiscoTope-3.0/)
- Mirror (BioLib): [https://biolib.com/DTU/DiscoTope-3/](https://biolib.com/DTU/DiscoTope-3/)

# Colab
To test the method out without installing it you can try this: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sMmzzno5fAeGb-r0D7R6lqo9Tld9LYiq)

# Quickstart guide

```bash
# Setup environment and install
conda create --name inverse python=3.9 -y
conda activate inverse
conda install -c pyg pyg -y
conda install -c conda-forge pip -y

git clone https://github.com/Magnushhoie/discotope3_web/
cd discotope3_web/
pip install .

# Unzip models to use
unzip models.zip

# 1. Predict single PDB (solved structure)
python discotope3/main.py --pdb_or_zip_file data/example_pdbs_solved/7c4s.pdb
# CPU only:
python discotope3/main.py --cpu_only --pdb_or_zip_file data/example_pdbs_solved/7c4s.pdb
```

# Repo contents

- [data](./data): Example input files, including test set
- [discotope3](./discotope3): Source code
- [output](./output): DiscoTope-3.0 output examples

# Installation guide

We highly recommend using an Ubuntu OS and Conda ([miniconda](https://docs.conda.io/en/main/miniconda.html) or [anaconda](https://www.anaconda.com/products/distribution)) for installing required dependencies.

Predictions are faster using a GPU and the recommended versions of pytorch, pytorch-geometric and cudatoolkit, but these exact versions are not required.

## For Linux & GPU with conda (recommended, ~2 mins) 

```bash
# Setup environment with conda
conda create -n inverse python=3.9
conda activate inverse
conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg -c conda-forge
conda install pip

# install pip dependencies
pip install .
```

## Linux & GPU with pip (~5 mins)
```bash
# install pip dependencies
pip install -r requirements_recommended.txt
pip install .
```

### Recommended system requirements
- GPU is optional. Recommended 16 GB ram, 2+ cores CPU.
- Linux operating system (e.g. Ubuntu 18.04), but works on MacOS
- [Python 3.9](https://www.python.org/downloads/)
- [Pytorch 1.11](https://pytorch.org/get-started/locally/)
- [cudatoolkit 11.3](https://anaconda.org/anaconda/cudatoolkit)
- [Pytorch geometric 2.0.4](https://github.com/pyg-team/pytorch_geometric)
- [Biopython](https://github.com/biopython/biopython)
- [Biotite](https://github.com/biotite-dev/biotite)
- [pandas](https://github.com/pandas-dev/pandas)
- [numpy](https://github.com/numpy/numpy)
- [py-xgboost-gpu](https://xgboost.readthedocs.io/en/stable/install.html)


## Running DiscoTope-3.0

DiscoTope-3.0 can predict a single PDB, a folder or ZIP file of PDBs, or fetch PDBs using their IDs from RCSB or AlphafoldDB to predict them.

On a common workstation with a GPU, predictions takes <1 second per PDB chain with ~ 15 seconds for loading needed libraries and model weight. 

Set the --struc_type parameter to 'solved' for experimentally solved structures (default) or 'alphafold' for modelled structures.

Note that DiscoTope-3.0 splits PDB structures into single chains before prediction, unless --multi_chain_mode is set.

```bash
# Unzip models
unzip models.zip

# Now select one of multiple options:

# 1. Predict single PDB (solved)
python discotope3/main.py --pdb_or_zip_file data/example_pdbs_solved/7c4s.pdb

# 2. Predict AlphaFold structure
python discotope3/main.py --pdb_or_zip_file data/example_pdbs_alphafold/7tdm_B.pdb --struc_type alphafold

# 3. Predict a folder of PDBs
python discotope3/main.py --pdb_dir data/example_pdbs_solved --out_dir output/example_pdbs_solved

# 4. Predict a ZIP file of PDBs
python discotope3/main.py --pdb_or_zip_file pdbs_in_zipfile.zip --out_dir output/pdbs_in_zipfile

# 5. Fetch PDBs from RCSB
python discotope3/main.py --list_file pdb_list_solved.txt --struc_type solved --out_dir output/pdb_list_solved

# 6. Fetch PDBs from Alphafolddb
python discotope3/main.py --list_file pdb_list_af2.txt --struc_type alphafold --out_dir output/pdb_list_af2

Predict B-cell epitope propensity on input protein PDB structures

optional arguments:
  -h, --help            show this help message and exit
  -f PDB_OR_ZIP_FILE, --pdb_or_zip_file PDB_OR_ZIP_FILE
                        Input file, either single PDB or compressed zip file with multiple PDBs
  --list_file LIST_FILE
                        File with PDB or Uniprot IDs, fetched from RCSB/AlphaFolddb
  --struc_type STRUC_TYPE
                        Structure type from file (solved | alphafold)
  --pdb_dir PDB_DIR     Directory with AF2 PDBs
  --out_dir OUT_DIR     Job output directory
  --models_dir MODELS_DIR
                        Path for .json files containing trained XGBoost ensemble
  --calibrated_score_epi_threshold CALIBRATED_SCORE_EPI_THRESHOLD
                        Calibrated-score threshold for epitopes [low 0.40, moderate (0.90), higher 1.50]
  --no_calibrated_normalization
                        Skip Calibrated-normalization of PDBs
  --check_existing_embeddings CHECK_EXISTING_EMBEDDINGS
                        Check for existing embeddings to load in pdb_dir
  --cpu_only            Use CPU even if GPU is available (default uses GPU if available)
  --max_gpu_pdb_length MAX_GPU_PDB_LENGTH
                        Maximum PDB length to embed on GPU (1000), otherwise CPU
  --multichain_mode     Predicts entire complexes, unsupported and untested
  --save_embeddings SAVE_EMBEDDINGS
                        Save embeddings to pdb_dir
  --web_server_mode     Flag for printing HTML output
  -v VERBOSE, --verbose VERBOSE
                        Verbose logging

```

# DiscoTope-3.0 output

DiscoTope-3.0 splits input PDBs into single-chain PDB files, then predict per-residue epitope propensity scores.
Outputs are saved in both PDB and CSV format.

The CSV output files contains per-residue outputs, with the following column headers:
- PDB ID and chain name
- Relative residue index (re-numbered from 1)
- Amino-acid residue, 1-letter
- DiscoTope-3.0 score (0.00 - 1.00)
- Predicted epitope (True or False), based on calibrated_score_epi_threshold (default 0.90)
- Relative surface accessibility (Shrake-Rupley, normalized using Sander scale)
- AlphaFold pLDDT score (0-100, set to 100 for non-AlphaFold structures)
- Chain length
- A binary feature set to 0 for solved and 1 for AlphaFold structures.

The PDB output files contain individual single chains with the B-factor column replaced with per-residue DiscoTope-3.0 scores (2nd right-most column). Note that the scores are multiplied by 100 as PDB files only allow 2 decimals of precision.

Example input PDB (see [7c4s.pdb](./data/example_pdbs_solved/7c4s.pdb)):
```bash
python discotope3/main.py --pdb_or_zip_file data/example_pdbs_solved/7c4s.pdb
```

Example output CSV (see [7c4s_A_discotope3.csv](./output/7c4s/output/7c4s_A_discotope3.csv)):
```text
pdb,res_id,residue,DiscoTope-3.0_score,rsa,pLDDTs,length,alphafold_struc_flag
7c4s_A,14,G,0.15186,0.80634,100,282,0
7c4s_A,15,Q,0.13953,0.45077,100,282,0
7c4s_A,16,E,0.23955,0.72919,100,282,0
```

Example output PDB (see [7c4s_A_discotope3.pdb](./output/7c4s/output/7c4s_A_discotope3.pdb)):
(Note DiscoTope-3.0 scores in the B-factor column)
```text
ATOM      1  N   GLY A  14     -16.773 -32.069  23.105  1.00 15.19           N  
ATOM      2  CA  GLY A  14     -15.595 -32.029  23.955  1.00 15.19           C  
ATOM      3  C   GLY A  14     -14.287 -31.844  23.204  1.00 15.19           C  
ATOM      4  O   GLY A  14     -13.284 -32.465  23.555  1.00 15.19           O  
```

## Reproduce test-set predictions (AlphaFold2 structures)

```bash
# Unzip AlphaFold2 test set
unzip data/test_set_af2.zip -d data/

# Run predictions on PDB folder
python discotope3/main.py \
--pdb_dir data/test_set_af2 \
--struc_type alphafold \
--out_dir output/test_set_af2
```

# Common issues

- No valid amino-acid backbone found: Occurs if only heteroatoms (non-amino acid residues) are found in the extracted chain. DiscoTope-3.0 requires full amino-acid backbone C, Ca and N atoms.
- PDBConstructionWarning regarding discontinuous chains: Indicates missing residue atoms in the input PDB file. May impact DiscoTope-3.0 performance (solved structures only)
- Biopython future deprecation warning: Benign Biopython library warning, does not impact predictions
- ESM regression weights missing warning: Benign fair-esm library warning, does not impact predictions

## Installation gcc or g++ errors, missing torch-scatter build ...
```bash
# Make sure gcc and g++ versions are updated, pybind11 is available
# torch-scatter should be listed with 'conda list' or 'pip list'

# With conda:
conda install -c conda-forge pybind11 gcc cxx-compiler

# With apt-get
sudo apt-get install gcc g++
pip install pybind11
```

# Citation
For usage of the package and associated manuscript, please cite according to the enclosed [citation.bib](./citation.bib).

# License

This source code is licensed under the Creative Commons license found in the [LICENSE](./LICENSE) file in the root directory of this source tree.
