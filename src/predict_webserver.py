# Discotope 3.0 predict on folder of PDBs script
# https://github.com/Magnushhoie/discotope3/

MAX_FILES = 50
MAX_FILE_SIZE_MB = 30

import logging

logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
log = logging.getLogger(__name__)

import glob
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import closing
# Set project path two levels up
from pathlib import Path
from typing import List
from zipfile import ZipFile

import biotite
import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import prody  # MH can rewrite with Biotite?
import xgboost as xgb
# from Bio import SeqIO
from sklearn import metrics

ROOT_PATH = str(os.path.dirname(os.getcwd()))

from argparse import ArgumentParser, RawTextHelpFormatter

import requests
from Bio.PDB import PDBIO, Select
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser

# Import make_dataset scripts
from make_dataset import Discotope_Dataset_web


def cmdline_args():
    # Make parser object
    usage = f"""
Options:
    1) PDB file (--pdb_or_zip_file)
    2) Zip file of PDBs (--pdb_or_zip_file)
    3) PDB directory (--pdb_dir)
    4) File with PDB ids on each line (--list_file)

# Predict on example PDBs in folder
python src/predict_webserver.py \
--pdb_dir data/test \
--struc_type solved \
--out_dir job_out/test

# Predict only on PDBs IDs specified in antigens.fasta entries
python src/predict_webserver.py \
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
        "--pdb_or_zip_file",
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
        "--list_id_type",
        help="PDB ID type (rcsb or uniprot)",
    )

    p.add_argument(
        "--pdb_dir",
        default="data/pdbs",
        help="Directory with AF2 PDBs",
        # type=lambda x: is_valid_path(p, x),
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
    num_models: int = 100,
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


def write_model_prediction_csvs_pdbs(
    models, dataset, out_dir, verbose: int = 0
) -> None:
    """Calculates predictions for dataset PDBs, saves output .csv and .pdb file"""

    for i, sample in enumerate(dataset):
        try:
            # Predict on antigen features
            y_hat = predict_using_models(models, sample["X_arr"])
            sample["y_hat"] = y_hat
            # sample["df_stats"].insert(4, "DiscoTope-3.0_score", y_hat)

            # Output CSV
            df_out = sample["df_stats"]
            df_out.insert(4, "DiscoTope-3.0_score", y_hat)

            # Round to 5 digits
            num_cols = ["DiscoTope-3.0_score", "rsa"]
            df_out[num_cols] = df_out[num_cols].applymap(lambda x: "{:.5f}".format(x))

            # Write to CSV
            outfile = f"{out_dir}/{sample['pdb_id']}_discotope3.csv"
            if verbose:
                log.info(f"Writing {sample['pdb_id']} to {outfile}")
            df_out.to_csv(outfile)

        except Exception as E:
            log.error(f"Unable to calculate/write predictions CSV: {E}")

        try:
            # Set B-factor field to DiscoTope-3.0 score
            atom_array = sample["PDB_biotite"]
            target_values = sample["y_hat"]
            atom_array = biotite.structure.renumber_res_ids(atom_array)
            atom_array.b_factor = target_values[atom_array.res_id - 1]

            # Write PDB
            outfile = f"{out_dir}/{sample['pdb_id']}_discotope3.pdb"
            if verbose:
                log.info(f"Writing {sample['pdb_id']} to {outfile}")
            strucio.save_structure(outfile, atom_array)

        except Exception as E:
            log.error(f"Unable to calculate/write predictions PDB: {E}")


def save_predictions_from_dataset(
    dataset: Discotope_Dataset_web,
    models: List["xgb.XGBClassifier"],
    out_dir: str,
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

        values = y_hat

        try:
            bfacs = ((values - values.min()) / (values.max() - values.min())) * 100
            struc = prody_set_bfactor(struc, bfacs)
            prody.writePDB(f"{out_dir}/{sample['pdb_id']}_{label}.pdb", struc)
            p = PDBParser(PERMISSIVE=1)
            structure = p.get_structure(
                f"{sample['pdb_id']}_{label}",
                f"{out_dir}/{sample['pdb_id']}_{label}.pdb",
            )
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(f"{out_dir}/{sample['pdb_id']}_{label}.tmp.cif", Clean_Chain(bfacs))
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
                    if previous_resid is None or previous_resid != int(entry[26:30]):
                        qa_metric = [" " for _ in range(24)]
                        auth_id = entry.split()[15]
                        # qa_metric[20:20+len(auth_id)] = auth_id
                        qa_metric[-4:] = entry[26:30]
                        qa_metric[0] = entry[22]
                        qa_metric[2:5] = entry[18:21]
                        qa_metric[6 : 6 + len(auth_id)] = auth_id
                        # qa_metric[6:10] = entry[26:30]
                        qa_metric[10] = "2"
                        qa_metric[12:17] = "{:.6f}".format(float(entry.split()[14]))[:5]
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


def save_pdb(pdb_name, pdb_path, out_prefix, score):

    p = PDBParser(PERMISSIVE=True)
    structure = p.get_structure(pdb_name, pdb_path)

    # chains = structure.get_chains()

    pdb_out = f"{out_prefix}.pdb"
    io_w_no_h = PDBIO()
    io_w_no_h.set_structure(structure)
    io_w_no_h.save(pdb_out, Clean_Chain(score))

    # cif_out = f"{out_prefix}.cif"
    # io = MMCIFIO()
    # io.set_structure(structure)
    # io.save(cif_out, Clean_Chain(score))


def fetch_and_process_from_list_file(list_file, out_dir):
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
        if args.list_id_type == "uniprot":
            URL = f"https://alphafold.ebi.ac.uk/files/AF-{prot_id}-F1-model_v4.pdb"
            score = None
        elif args.list_id_type == "rcsb":
            URL = f"https://files.rcsb.org/download/{prot_id}.pdb"
            score = 100
        else:
            log.error(f"Structure ID was of unknown type {args.list_id_type}")
            sys.exit(0)

        response = requests.get(URL)
        if response.status_code == 200:
            with open(f"{out_dir}/temp", "wb") as f:
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

        save_pdb(f"{prot_id}", f"{out_dir}/temp", f"{out_dir}/{prot_id}", score)


def true_if_zip(infile):
    """Returns True if file header bits are zip file"""
    with open(infile, "rb") as fb:
        header_bits = fb.read(4)
    return header_bits == b"PK\x03\x04"


def check_valid_input(args):
    """Checks for valid arguments"""

    # Check input arguments
    if not (args.pdb_or_zip_file or args.pdb_dir or args.list_file):
        log.error(
            f"""Please choose one of:
        1) PDB file (--pdb_or_zip_file)
        2) Zip file with PDBs (--pdb_or_zip_file)
        3) PDB directory (--pdb_dir)
        4) File with PDB ids on each line (--list_file)
        """
        )
        sys.exit(0)

    if args.list_file and not args.list_id_type:
        log.error(f"Must provide list_id_type (rcsb or uniprot) with list_file")
        sys.exit()

    if args.struc_type not in ["solved", "alphafold"]:
        log.error(
            f"--struc_type flag invalid, must be solved or alphafold. Found {args.struc_type}"
        )
        sys.exit(0)

    if (
        (args.pdb_dir and args.list_file)
        or (args.pdb_dir and args.pdb_or_zip_file)
        or (args.list_file and args.pdb_or_zip_file)
    ):
        log.error(
            f"Please choose only one of flags: pdb_dir, list_file or pdb_or_zip_file"
        )
        sys.exit(0)

    if args.pdb_dir and args.pdb_or_zip_file:
        log.error(f"Both pdb_dir and list_file flags set, please chooose one")
        sys.exit(0)

    # Check ZIP max-size, number of files
    if args.pdb_or_zip_file:
        size_mb = os.stat(args.pdb_or_zip_file).st_size / (1024 * 1024)
        if size_mb > MAX_FILES:
            log.error(f"Max file-size {MAX_FILE_SIZE_MB} MB, found {round(size_mb)} MB")
            sys.exit(0)

        if true_if_zip(args.pdb_or_zip_file):
            with closing(ZipFile(args.pdb_or_zip_file)) as archive:
                file_count = len(archive.infolist())
                file_names = archive.namelist()

            # Check number of files in zip
            if file_count > MAX_FILES:
                log.error(f"Max number of files {file_count}, found {file_count}")
                sys.exit(0)

            # Check filenames end in .pdb
            name = file_names[0]
            if os.path.splitext(name)[-1] != ".pdb":
                log.error(
                    f"Ensure all ZIP content file-names end in .pdb, found {name}"
                )
                sys.exit(0)


def write_predictions_zip_file(
    predictions_dir: str, out_dir: str, verbose: int = 0
) -> str:
    """Returns filepath for compressed predictions zip stored in out_dir"""
    timestamp = time.strftime("%Y%m%d%H%M")
    outbasename = f"discotope3_{timestamp}"

    if args.verbose:
        log.info(f"Compressing ZIP file {outbasename}.zip")

    shutil.make_archive(
        f"{outbasename}",
        "zip",
        root_dir=predictions_dir,
    )

    return f"{outbasename}.zip"


def get_basename_no_ext(filepath):
    """
    Returns file basename excluding extension,
    e.g. dir/5kja_A.pdb -> 5kja_A
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def get_directory_basename_dict(directory: str, glob_ext: str) -> dict:
    """
    Returns dict of file basenames ending with glob_ext
    E.g. get_directory_basename_dict(pdb_dir, "*.pdb")
    Returns: {'7k7i_B': '../data/mini/7k7i_B.pdb'}
    """
    dir_list = glob.glob(f"{directory}/{glob_ext}")
    return {get_basename_no_ext(fp): fp for fp in dir_list}


def check_missing_pdb_csv_files(in_dir, out_dir) -> None:
    """Reports missing CSV and PDB file in out_dir, per PDB file in in_dir"""

    # Get basenames of input PDBs and output PDB/CSV files
    in_pdb_dict = get_directory_basename_dict(in_dir, "*.pdb")
    out_dict = get_directory_basename_dict(out_dir, "*.[pdb|csv]*")

    # Remove _discotope3 extension before comparison
    out_dict = {re.sub(r"_discotope3$", "", k): v for k, v in out_dict.items()}

    # Log which input files are not found in output
    missing_pdbs = in_pdb_dict.keys() - out_dict.keys()
    if len(missing_pdbs) > 0:
        log.error(f"Error: Failed processing PDBs {''.join(list(missing_pdbs))}")


def main(args):
    """Main function"""

    # Error messages if invalid input
    check_valid_input(args)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as pdb_or_tempdir:

        # 1. Download PDBs from RCSB or AlphaFoldDB
        if args.list_file:
            log.info(f"Fetching PDBs")
            fetch_and_process_from_list_file(
                args.list_file, pdb_or_tempdir, pdb_or_tempdir
            )

        # 2. Unzip if ZIP, else single PDB
        if args.pdb_or_zip_file:
            if true_if_zip(args.pdb_or_zip_file):
                log.info(f"Unzipping PDBs")
                zf = ZipFile(args.pdb_or_zip_file)
                zf.extractall(pdb_or_tempdir)

            # 3. If single PDB, copy to tempdir
            else:
                shutil.copy(args.pdb_or_zip_file, pdb_or_tempdir)

        # 4. Load from PDB folder
        if args.pdb_dir:
            pdb_or_tempdir = args.pdb_dir

        # Embed and predict
        log.info(f"Pre-processing PDBs")
        dataset = Discotope_Dataset_web(
            pdb_or_tempdir, structure_type=args.struc_type, verbose=args.verbose
        )

        # Predict and save
        log.info(f"Loading XGBoost ensemble")
        models = load_models(args.models_dir, num_models=2)  # MH

        log.info(f"Writing prediction .csv and .pdb files")
        write_model_prediction_csvs_pdbs(
            models, dataset, out_dir=args.out_dir, verbose=args.verbose
        )

        # Zip output folder
        log.info(f"Writing predictions CSV file")
        # temp_id = "/".join(args.out_dir.rsplit("/", 2)[1:])
        # job_out_dir = f"/services/DiscoTope-3.0/tmp/{temp_id}"
        out_zip = write_predictions_zip_file(
            predictions_dir=pdb_or_tempdir, out_dir=args.out_dir, verbose=args.verbose
        )

        # Check which files failed
        check_missing_pdb_csv_files(pdb_or_tempdir, args.out_dir)
        log.info(f"Done!")
        asd

        # HTML printing
        examples = """<script type="text/javascript">const examples = ["""
        print("<h2>Output download</h2>")
        print(
            f'<a href="{out_zip}"><p>Download DiscoTope-3.0 prediction results as zip</p></a>'
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
            outpdb = f"{temp_id}/{sample['pdb_id']}_discotope3.pdb"
            outcsv = f"{temp_id}/{sample['pdb_id']}_discotope3.csv"

            examples += "{"
            examples += f"id:'{sample['pdb_id']}',url:'https://services.healthtech.dtu.dk/services/DiscoTope-3.0/tmp/{outpdb}',info:'Structure {i+1}'"
            examples += "},"

            style = 'style="margin-top:2em;"' if i > 0 else ""
            print(
                f"<h3 {style}>{sample['pdb_id']} (chains {'/'.join(pdb_chains_dict[sample['pdb_id']])})</h3>"
            )
            print(
                f'<a href="/services/DiscoTope-3.0/tmp/{outpdb}"><p>Download PDB w/ DiscoTope-3.0 prediction scores</p></a>'
            )
            print(
                f'<a href="/services/DiscoTope-3.0/tmp/{outcsv}"><p>Download CSV</p></a>'
            )

        print("</div></div></div>")
        examples += "];</script>"
        print(examples)


if __name__ == "__main__":

    args = cmdline_args()
    logging.basicConfig(
        filename=f"{args.out_dir}/dt3.log",
        encoding="utf-8",
        level=logging.INFO,
        format="[{asctime}] {message}",
        style="{",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)
    log.info("Predicting PDBs using Discotope-3.0")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.pdb_dir, exist_ok=True)

    try:
        main(args)
    except Exception as E:
        log.exception(
            "Prediction encountered an unexpected error. This is likely a bug in the server software."
        )
