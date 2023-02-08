## Abstract

DiscoTope-3.0: Improved B-cell epitope prediction using AlphaFold2 modeling and inverse folding latent representations
Magnus Haraldson Høie, Frederik Steensgaard Gade, Julie Maria Johansen, Charlotte Würtzen, Ole Winther, Morten Nielsen, Paolo Marcatili. bioRxiv (Feb 2023). doi: [10.1101/2023.02.05.527174](https://www.biorxiv.org/content/10.1101/2023.02.05.527174v1)

Accurate computational identification of B-cell epitopes is crucial for the development of vaccines, therapies, and diagnostic tools. Structure-based prediction methods generally outperform sequence-based models, but are limited by the availability of experimentally solved structures. Here, we present DiscoTope-3.0, a B-cell epitope prediction tool that exploits inverse folding representations from solved or AlphaFold-predicted structures. On independent datasets, the method demonstrates improved performance on both linear and non-linear epitopes with respect to current state-of-the-art algorithms. Most notably, our tool maintains high predictive performance across solved and predicted structures, alleviating the need for experiments and extending the general applicability of the tool by more than 4 orders of magnitude. DiscoTope-3.0 is available as a web server and downloadable package, processing up to 50 structures per submission. The web server interfaces with RCSB and AlphaFoldDB, enabling large-scale prediction on all currently cataloged proteins. DiscoTope-3.0 is available as a web server at [https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0](https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0)

## Setup

```bash
# Full training, validation and test dataset available at:
https://services.healthtech.dtu.dk/service.php?DiscoTope-3.0

# Unzip models
unzip models.zip

# Setup environment with conda
conda create -n inverse python=3.9
conda activate inverse
conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch   ## very important to specify pytorch package!
conda install pyg -c pyg -c conda-forge ## very important to make sure pytorch and cuda versions not being changed
conda install pip
pip install biotite
pip install git+https://github.com/facebookresearch/esm.git

# further installation: pytorch lightning, pandas, pyyaml, fastpdb, wandb, xgboost, py-xgboost-gpu
pip install -r requirements.txt
```

## Usage 
```bash
# Predict on example PDBs in data folder
python src/predict_webserver.py \
--pdb_dir data/ \
--struc_type solved \
--out_dir job_out/data

# Fetch PDBs from list file from AlphaFoldDB
python src/predict_webserver.py \
--list_file data/af2_list_uniprot.txt \
--struc_type alphafold \
--out_dir job_out/af2_list_uniprot

# See more options
python automate.py
```

