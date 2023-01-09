# Discotope 3.0 predict on folder of PDBs script
# https://github.com/Magnushhoie/discotope3/

MAX_FILES = 50

import glob
import logging
import os
import sys
# Set project path two levels up
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import prody  # MH can rewrite with Biotite?
import xgboost as xgb
from Bio import SeqIO
from sklearn import metrics

ROOT_PATH = str(os.path.dirname(os.getcwd()))

from argparse import ArgumentParser, RawTextHelpFormatter

import requests
from Bio.PDB import PDBIO, Select
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser

# Import make_dataset scripts
from make_dataset import (Discotope_Dataset, embed_pdbs_IF1,
                          save_fasta_from_pdbs)


def cmdline_args():
    # Make parser object
    usage = f"""
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
    """
    p = ArgumentParser(
        description="Predict Discotope-3.0 score on folder of input AF2 PDBs",
        formatter_class=RawTextHelpFormatter,
        usage=usage,
    )

    def is_valid_path(parser, arg):
        if not os.path.exists(arg):
            parser.error(f"Path {arg} does not exist!")
        else:
            return arg

    p.add_argument(
        "-f",
        "--in_file",
        dest="pdb_or_zip_file",
        help="Input file, either single PDB or compressed zip file with multiple PDBs",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--list_file",
        help="File with PDB or Uniprot IDs, fetched from RCSB/AlphaFolddb",
    )

    p.add_argument(
        "--struc_type",
        required=True,
        help="Structure type from file (solved | alphafold)",
    )

    p.add_argument(
        "--id_type",
        help="ID type (rcsb or uniprot)",
    )

    p.add_argument(
        "--pdb_dir",
        default="data/pdbs",
        help="Directory with AF2 PDBs",
        #type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--out_dir",
        default="job_out/job1",
        help="Job output directory",
    )

    p.add_argument(
        "--models_dir",
        default="models/",
        help="Path for .json files containing trained XGBoost ensemble",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--skip_embeddings",
        help="Skip ESM-IF1 embedding step (False)",
        default=False,
        type=bool,
    )

    p.add_argument(
        "--overwrite_embeddings",
        dest="overwrite_embeddings",
        help="Recreate PDB ESM-IF1 embeddings even if existing",
        default=False,
    )
    p.add_argument("-v", "--verbose", type=int, default=0, help="Verbose logging")

    return p.parse_args()


def get_percentile_score(
    df: pd.DataFrame,
    col: str,
):
    """Find mean predicted epitope rank percentile score"""
    epitopes = df["epitope"].astype(bool)
    c = df[col][epitopes].mean()
    c_percentile = (c > df[col]).mean()

    return c_percentile, c


def load_models(
    models_dir: str = "models/final__solved_pred/",
    num_models: int = 40,
    verbose: int = 1,
) -> List["xgb.XGBClassifier"]:
    """Loads saved XGBoostClassifier files containing model weights, returns list of XGBoost models"""
    import xgboost as xgb

    # Search for model files
    model_files = list(Path(models_dir).glob(f"XGB_*_of_*.json"))

    if len(model_files) < 1:
        log.error(f"Error: no files found in {models_dir}")
        raise Exception

    # Initialize new XGBoostClassifier and load model weights
    log.info(
        f"Loading {num_models} / {len(model_files)} XGBoost models from {models_dir}"
    )

    models = []
    for fp in model_files[:num_models]:
        m = xgb.XGBClassifier()
        m.load_model(str(fp))
        models.append(m)

    return models


def predict_using_models(
    models: List[xgb.XGBClassifier],
    X: np.array,
) -> np.array:
    """Returns np.array of predictions averaged from ensemble of XGBoost models"""

    def predict_PU_prob(X, estimator, prob_s1y1):
        """
        Predict probability using trained PU classifier,
        weighted by prob_s1y1 = c
        """
        predicted_s = estimator.predict_proba(X)

        if len(predicted_s.shape) != 1:
            predicted_s = predicted_s[:, 1]

        return predicted_s / prob_s1y1

    # Predict
    y_hat = np.zeros(len(X))
    for model in models:
        y_hat += predict_PU_prob(X, model, prob_s1y1=1)

    y_hat = y_hat / len(models)
    return y_hat


def get_auc(y_true, y_pred, sig_dig=5):
    """Returns AUC"""
    auc = np.round(metrics.roc_auc_score(y_true, y_pred), sig_dig)
    return auc


def save_predictions_from_dataset(
    dataset: Discotope_Dataset,
    models: List["xgb.XGBClassifier"],
    out_dir: str,
    calculate_struc_propensity_flag: bool = False,
    verbose: int = 0,
) -> None:
    """Loads models, predicts on dataset and saves to out_dir"""

    def prody_set_bfactor(
        struc: prody.atomic.atomgroup.AtomGroup,
        res_bfactors: np.array,
    ):

        # Set pLDDTs by residue indices
        idxs = struc.getResindices()
        struc.setBetas(res_bfactors[idxs])

        return struc

    dfs_list = []
    for i, sample in enumerate(dataset):

        # Predict on antigen features
        y_hat = predict_using_models(models, sample["X_arr"])

        # Add predictions to antigen dataframe
        df_pdb = sample["df_stats"].copy()
        df_pdb.insert(loc=3, column="Discotope-3.0_score", value=y_hat)
        df_pdb["epitope"] = df_pdb["residue"].apply(lambda s: s.isupper())

        # Write CSV and append to list
        outfile = f"{out_dir}/{sample['pdb_id']}.csv"
        log.info(
            f"{i+1} / {len(dataset)}: Saving predictions for {sample['pdb_id']} ({len(y_hat)} residues) to {outfile}"
        )
        df_pdb.to_csv(outfile, index=False)
        dfs_list.append(df_pdb)

        log.info(f"Saving PDBs")
        # Write output PDBs with B-factors set to predictions
        struc = prody.parsePDB(sample["pdb_fp"])

        d = {
            # "epi": sample["y_arr"],
            "raw": y_hat,
        }

        for label, values in d.items():
            try:
                bfacs = ((values - values.min()) / (values.max() - values.min())) * 100
                struc = prody_set_bfactor(struc, bfacs)
                prody.writePDB(f"{out_dir}/{sample['pdb_id']}_{label}.pdb", struc)
                # print(f"Outfile: {out_dir}/{sample['pdb_id']}_{label}.pdb")
                p = PDBParser(PERMISSIVE=1)
                structure = p.get_structure(
                    f"{sample['pdb_id']}_{label}",
                    f"{out_dir}/{sample['pdb_id']}_{label}.pdb",
                )
                io = MMCIFIO()
                io.set_structure(structure)
                io.save(
                    f"{out_dir}/{sample['pdb_id']}_{label}.tmp.cif", Clean_Chain(bfacs)
                )
                with open(
                    f"{out_dir}/{sample['pdb_id']}_{label}.tmp.cif", "r"
                ) as infile, open(
                    f"{out_dir}/{sample['pdb_id']}_{label}.cif", "w"
                ) as outfile:
                    outfile.write(infile.readline())
                    print(
                        "#",
                        "loop_",
                        "_ma_qa_metric.id",
                        "_ma_qa_metric.mode",
                        "_ma_qa_metric.name",
                        "_ma_qa_metric.software_group_id",
                        "_ma_qa_metric.type",
                        "1 global pLDDT 1 pLDDT",
                        "2 local  pLDDT 1 pLDDT",
                        sep="\n",
                        file=outfile,
                    )
                    print(
                        "#",
                        "loop_",
                        "_ma_qa_metric_local.label_asym_id",
                        "_ma_qa_metric_local.label_comp_id",
                        "_ma_qa_metric_local.label_seq_id",
                        "_ma_qa_metric_local.metric_id",
                        "_ma_qa_metric_local.metric_value",
                        "_ma_qa_metric_local.model_id",
                        "_ma_qa_metric_local.ordinal_id",
                        sep="\n",
                        file=outfile,
                    )

                    info_header = list()
                    for i in range(20):
                        info_header.append(infile.readline().strip())

                    atoms_cif = [x.strip() for x in infile.readlines()][:-1]
                    previous_resid = None

                    for entry in atoms_cif:
                        if previous_resid is None or previous_resid != int(
                            entry[26:30]
                        ):
                            qa_metric = [" " for _ in range(24)]
                            auth_id = entry.split()[15]
                            # qa_metric[20:20+len(auth_id)] = auth_id
                            qa_metric[-4:] = entry[26:30]
                            qa_metric[0] = entry[22]
                            qa_metric[2:5] = entry[18:21]
                            qa_metric[6 : 6 + len(auth_id)] = auth_id
                            # qa_metric[6:10] = entry[26:30]
                            qa_metric[10] = "2"
                            qa_metric[12:17] = "{:.6f}".format(
                                float(entry.split()[14])
                            )[:5]
                            qa_metric[18] = "1"
                            previous_resid = int(entry[26:30])
                            print("".join(qa_metric), file=outfile)

                    for info in info_header:
                        print(info, file=outfile)
                    for entry in atoms_cif:
                        print(entry, file=outfile)
                    print("#", file=outfile)

            except Exception as E:
                log.error(f"Unable to write prediction PDB: {E}")

    # Merge and calculate performance
    df_all = pd.concat(dfs_list, ignore_index=False)

    # y_true = df_all["residue"].apply(lambda x: str(x).isupper())
    y_hat = df_all["Discotope-3.0_score"]
    # log.info(f"Merged, AUC {get_auc(y_true, y_hat):.5f}")

    return df_all


def merge_prediction_csvs(csv_dir: str, outfile: str):
    """Merge discotope CSV files to single and save to outdir"""

    # Discotope
    # csv_dir = "../webserver/data/dt3"
    csv_files = list(Path(csv_dir).glob("*.csv"))
    log.info(f"DT3 to DT3: Found {len(csv_files)} CSV files in {csv_dir}")

    # Combine and save
    dt3_df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    df_dt3 = pd.concat(dt3_df_list, ignore_index=False)
    df_dt3.reset_index(drop=True)

    log.info(f"Saving Discotope-3.0 merged CSV to {outfile}")
    df_dt3.to_csv(outfile, index=False)


class Clean_Chain(Select):
    def __init__(self, score):
        self.score = score
        if score is None:
            self.const_score = None
        elif isinstance(score, (int, float)):
            self.const_score = True
        else:
            self.const_score = False
        self.init_resid = None
        self.prev_resid = None
        self.letter_correction = 0
        self.letter = " "
        self.prev_letter = " "

    # Clean out non-heteroatoms
    # https://stackoverflow.com/questions/25718201/remove-heteroatoms-from-pdb
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0

    def accept_atom(self, atom):
        if self.const_score is None:
            pass
        elif self.const_score:
            atom.set_bfactor(self.score)
        else:
            self.letter = atom.get_full_id()[3][2]
            if atom.get_full_id()[3][2] not in (self.prev_letter, " "):
                log.info(
                    f"A residue with lettered numbering was found ({atom.get_full_id()[3][1]}{atom.get_full_id()[3][2]}). This may mess up visualisation."
                )
                self.letter_correction += 1
            self.prev_letter = self.letter

            res_id = atom.get_full_id()[3][1] + self.letter_correction

            if self.init_resid is None:
                self.init_resid = res_id

            if self.prev_resid is not None and res_id - self.prev_resid > 1:
                self.init_resid += res_id - self.prev_resid - 1

            self.prev_resid = res_id

            atom.set_bfactor(self.score[res_id - self.init_resid])
        return True


def save_pdb_and_cif(pdb_name, pdb_path, out_prefix, score):

    p = PDBParser(PERMISSIVE=True)
    structure = p.get_structure(pdb_name, pdb_path)

    # chains = structure.get_chains()

    pdb_out = f"{out_prefix}.pdb"
    cif_out = f"{out_prefix}.cif"

    io_w_no_h = PDBIO()
    io_w_no_h.set_structure(structure)
    io_w_no_h.save(pdb_out, Clean_Chain(score))

    io = MMCIFIO()
    io.set_structure(structure)
    io.save(cif_out, Clean_Chain(score))


def fetch_and_process_from_list_file(list_file, out_dir, tmp_dir):
    """Fetch and process PDB chains/UniProt entries from list input"""

    with open(list_file, "r") as f:
        pdb_list = sorted(set([line.strip() for line in f.readlines()]))

    if len(pdb_list) == 0:
        log.error("No IDs found in list.")
        sys.exit(0)
    elif len(pdb_list) > MAX_FILES:
        log.error(
            f"A maximum of {MAX_FILES} PDB IDs can be processed at one time ({len(pdb_list)} IDs found)."
        )
        sys.exit(0)

    for prot_id in pdb_list:
        if args.id_type == "uniprot":
            URL = f"https://alphafold.ebi.ac.uk/files/AF-{prot_id}-F1-model_v4.pdb"
            score = None
        elif args.id_type == "rcsb":
            URL = f"https://files.rcsb.org/download/{prot_id}.pdb"
            score = 100
        else:
            log.error(f"Structure ID was of unknown type {args.id_type}")
            sys.exit(0)

        response = requests.get(URL)
        if response.status_code == 200:
            with open(f"{tmp_dir}/temp.pdb", "wb") as f:
                f.write(response.content)
        elif response.status_code == 404:
            log.error(f"File with the given ID could not be found (url: {URL}).")
            log.error("Maybe you selected the wrong ID type or misspelled the ID.")
            sys.exit(0)
        elif response.status_code in (408, 504):
            log.error(
                f"Request timed out with error code {response.status_code} (url: {URL})."
            )
            log.error(
                "Try to download the structure(s) locally from the given database and upload as pdb or zip."
            )
            sys.exit(0)
        else:
            log.error(
                f"Received status code {response.status_code}, when trying to fetch file from {URL}"
            )
            sys.exit(0)

        save_pdb_and_cif(
            f"{prot_id}", f"{tmp_dir}/temp.pdb", f"{out_dir}/{prot_id}", score
        )


def main(args):
    """Main function"""

    if not (args.pdb_or_zip_file or args.list_file):
        log.error(f"No input PDB or list file given")
        sys.exit(0)

    # Make sure out_dir exists
    if args.pdb_or_zip_file:
        zip_file = False
        with open(args.pdb_or_zip_file, "rb") as fb:
            header_bits = fb.read(4)
            if header_bits == b"PK\x03\x04":
                zip_file = True

        if zip_file:
            log.info(f"Reading ZIP file: {os.path.basename(args.pdb_or_zip_file)}")
            os.system(f"unzip {args.pdb_or_zip_file} -d {args.pdb_dir} > /dev/null 2>&1")

            pdb_files_from_zip = glob.glob(f"{args.pdb_dir}/*.pdb")
            if len(pdb_files_from_zip) == 0:
                log.error("No .pdb files found in zip file. Ensure files end in .pdb, and are not contained in folders/subfolder.")
                sys.exit(0)
            elif len(pdb_files_from_zip) > MAX_FILES:
                log.error(
                    f"A maximum of {MAX_FILES} PDB files can be processed at one time ({len(pdb_files_from_zip)} files found)."
                )
                sys.exit(0)

            for pdb_file in pdb_files_from_zip:
                pdb_prefix = pdb_file.rsplit(".", 1)[0]
                pdb_name = pdb_prefix.rsplit("/", 1)[-1]
                if args.struc_type == "solved":
                    save_pdb_and_cif(pdb_name, pdb_file, pdb_prefix, None)
                elif args.struc_type == "alphafold":
                    save_pdb_and_cif(pdb_name, pdb_file, pdb_prefix, 100)
                else:
                    log.error(
                        f"Structure type was of unexpected type {args.struc_type}."
                    )
                    sys.exit(0)
                os.remove(pdb_file)
        else:
            if args.struc_type == "solved":
                save_pdb_and_cif(
                    f"Custom upload", args.pdb_or_zip_file, f"{args.pdb_dir}/upload", None
                )
            elif args.struc_type == "alphafold":
                save_pdb_and_cif(
                    f"Custom upload", args.pdb_or_zip_file, f"{args.pdb_dir}/upload", 100
                )
            else:
                log.error(f"Structure type was of unexpected type {args.struc_type}.")
                sys.exit(0)

    if args.list_file:
        fetch_and_process_from_list_file(args.list_file, args.pdb_dir, args.out_dir)

    log.info(f"Creating FASTA file from PDB sequences: {args.out_dir}/pdbs.fasta")
    save_fasta_from_pdbs(args.pdb_dir, args.out_dir)
    input_fasta = f"{args.out_dir}/pdbs.fasta"

    # Load provided/created FASTA file and write again to args.out_dir
    fasta_dict = SeqIO.to_dict(SeqIO.parse(input_fasta, "fasta"))
    log.info(f"Read {len(fasta_dict)} entries from {input_fasta}")

    with open(f"{args.out_dir}/pdbs.fasta", "w") as out_handle:
        SeqIO.write(fasta_dict.values(), out_handle, "fasta")

    # Create IF-1 embeddings unless argument to skip set
    if not args.skip_embeddings:
        log.info(f"Loading ESM-IF1 to embed PDBs")
        embed_pdbs_IF1(
            pdb_dir=args.pdb_dir,
            out_dir=args.pdb_dir,
            input_fasta=fasta_dict,
            struc_type=args.struc_type,
            overwrite_embeddings=args.overwrite_embeddings,
        )

    log.info(f"Pre-processing PDBs")
    dataset = Discotope_Dataset(
        fasta_dict, args.pdb_dir, IF1_dir=args.pdb_dir, verbose=args.verbose
    )

    if len(dataset) == 0:
        # TODO: Specify what happened
        log.error("No valid file was supplied.")
        sys.exit(0)

    log.info(f"Loading XGBoost ensemble")
    models = load_models(args.models_dir)

    log.info(f"Predicting on PDBs")
    save_predictions_from_dataset(dataset, models, args.out_dir, verbose=args.verbose)

    log.info(f"Done!")

    examples = """<script type="text/javascript">const examples = ["""

    print("<h2>Output download</h2>")

    temp_id = "/".join(args.out_dir.rsplit("/", 2)[1:])

    os.system(f"cd {args.out_dir} && zip archive *epi.pdb *.csv > /dev/null 2>&1")
    print(
        f'<a href="/services/DiscoTope-3.0/tmp/{temp_id}/archive.zip"><p>Download DiscoTope-3.0 prediction results as zip</p></a>'
    )

    print(
        """<div class="wrap-collabsible">
        <input id="collapsible" class="toggle" type="checkbox">
        <label for="collapsible" class="lbl-toggle">Individual result downloads</label>
        <div class="collapsible-content">
        <div class="content-inner">
        """
    )

    pdb_chains_dict = dict()
    for pdb_w_chain_id in fasta_dict.keys():
        pdb, chain = pdb_w_chain_id.rsplit("_", 1)
        pdb_chains_dict[pdb] = chain

    for i, sample in enumerate(dataset):
        outpdb = f"{temp_id}/{sample['pdb_id']}_raw.pdb"
        outcsv = f"{temp_id}/{sample['pdb_id']}.csv"
        outcif = f"{temp_id}/{sample['pdb_id']}_raw.cif"

        examples += "{"
        examples += f"id:'{sample['pdb_id']}',url:'https://services.healthtech.dtu.dk/services/DiscoTope-3.0/tmp/{outcif}',info:'Structure {i+1}'"
        examples += "},"

        style = 'style="margin-top:2em;"' if i > 0 else ""
        print(
            f"<h3 {style}>{sample['pdb_id']} (chains {'/'.join(pdb_chains_dict[sample['pdb_id']])})</h3>"
        )
        print(
            f'<a href="/services/DiscoTope-3.0/tmp/{outpdb}"><p>Download PDB w/ DiscoTope-3.0 prediction scores</p></a>'
        )
        print(f'<a href="/services/DiscoTope-3.0/tmp/{outcsv}"><p>Download CSV</p></a>')

    print("</div></div></div>")
    examples += "];</script>"
    print(examples)

def check_valid_input(args):
    """ Checks for valid arguments """

    size_mb = os.stat(args.in_file) / (1024 * 1024)
    print(size_mb)

    asd


if __name__ == "__main__":
    # print("<h3>Debugging output</h3>")

    args = cmdline_args()
    logging.basicConfig(
        filename=f"{args.out_dir}/dt3.log",
        encoding="utf-8",
        level=logging.INFO,
        format="[{asctime}] {message}",
        style="{",
    )
    log = logging.getLogger(__name__)
    log.info("Predicting PDBs using Discotope-3.0")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.pdb_dir, exist_ok=True)

    main(args)
