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

