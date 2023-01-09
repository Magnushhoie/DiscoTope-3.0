
```bash
# Predict on example PDBs in folder
python src/predict_pdb.py \
--pdb_dir data/test \
--struc_type solved \
--out_dir job_out/test

# Predict only on PDBs IDs specified in antigens.fasta entries
python src/predict_pdb.py \
--fasta data/test.fasta \
--pdb_dir pdbs_embeddings \
--out_dir job_out/test
```
