## Setup

```bash
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
# Predict zip file of solved PDBs
python src/predict_webserver.py --in_file data/single_solved.zip --struc_type solved --verbose 2

# Predict list file, Alphafold UNIPROT IDs
python src/predict_webserver.py --list_file af2_list_uniprot --list_id uniprot --struc_type alphafold --out_dir job_out/test_af2

# Predict list file, RCSB solved IDs
python src/predict_webserver.py --list_file solved_list_rcsb --list_id rcsb --struc_type solved --out_dir job_out/test_solved

# See more options
python automate.py
```

