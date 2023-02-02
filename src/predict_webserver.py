# Discotope 3.0 predict on folder of PDBs script
# https://github.com/Magnushhoie/discotope3/

MAX_FILES = 50
MAX_FILE_SIZE_MB = 30

import logging

logging.basicConfig(level=logging.ERROR, format="[{asctime}] {message}", style="{")
log = logging.getLogger(__name__)

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import closing
# Set project path two levels up
from pathlib import Path
from typing import List
from zipfile import ZipFile

import biotite
import biotite.structure.io as strucio
import numpy as np
import xgboost as xgb

ROOT_PATH = str(os.path.dirname(os.getcwd()))

from argparse import ArgumentParser, RawTextHelpFormatter

import requests
from Bio.PDB import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser

# Import make_dataset scripts
from make_dataset import Discotope_Dataset_web


def cmdline_args():
    # Make parser object
    usage = rf"""
Options:
    1) PDB file (--pdb_or_zip_file)
    2) Zip file of PDBs (--pdb_or_zip_file)
    3) PDB directory (--pdb_dir)
    4) File with PDB ids on each line (--list_file)

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
        "--web_server_mode",
        default=False,
        help="Web server mode (True | (False)) prints HTML output",
    )

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
        required=False,  # Only needed for file input, not list
        default="alphafold",
        help="Structure type from file (solved | alphafold)",
    )

    p.add_argument(
        "--pdb_dir",
        help="Directory with AF2 PDBs",
        type=lambda x: is_valid_path(p, x),
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
        "--check_existing",
        default=False,
        help="Check for existing embeddings to load in pdb_dir",
    )

    p.add_argument(
        "--save_embeddings",
        default=False,
        help="Save embeddings to pdb_dir",
    )

    p.add_argument("-v", "--verbose", type=int, default=0, help="Verbose logging")

    # Print help if no arguments
    if len(sys.argv) == 1:
        p.print_help(sys.stderr)
        sys.exit(1)

    return p.parse_args()


def load_models(
    models_dir: str,
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


def write_model_prediction_csvs_pdbs(
    models, dataset, pdb_or_tempdir, out_dir, verbose: int = 0
) -> None:
    """Calculates predictions for dataset PDBs, saves output .csv and .pdb files"""

    os.makedirs(f"{out_dir}/download", exist_ok=True)

    for i, sample in enumerate(dataset):
        try:
            # Predict on antigen features
            y_hat = predict_using_models(models, sample["X_arr"])
            sample["y_hat"] = y_hat * 100

            # Output CSV
            df_out = sample["df_stats"]
            df_out.insert(3, "DiscoTope-3.0_score", y_hat)

            # Round to 5 digits
            num_cols = ["DiscoTope-3.0_score", "rsa"]
            df_out[num_cols] = df_out[num_cols].applymap(lambda x: "{:.5f}".format(x))

            # Write to CSV
            outfile = f"{out_dir}/{sample['pdb_id']}_discotope3.csv"
            if verbose:
                log.info(
                    f"Writing {sample['pdb_id']} ({i+1}/{len(dataset)}) to {outfile}"
                )
            df_out.to_csv(outfile, index=False)

        except Exception as E:
            log.error(
                f"PDB {sample['pdb_id']} {i+1}/{len(dataset)}: Unable to write predictions CSV: {E}"
            )
            traceback.print_exc()

        try:
            # Set B-factor field to DiscoTope-3.0 score
            atom_array = sample["PDB_biotite"]
            values = sample["y_hat"]
            # target_values = ((values - values.min()) / (values.max() - values.min())) * 100

            # Get relative indices starting from 1
            atom_array_renum = biotite.structure.renumber_res_ids(atom_array, start=1)
            atom_array.b_factor = values[atom_array_renum.res_id - 1]

            # Write PDB
            outfile = f"{out_dir}/{sample['pdb_id']}_discotope3.pdb"
            if verbose:
                log.info(
                    f"Writing {sample['pdb_id']} ({i+1}/{len(dataset)}) to {outfile}"
                )

            HEADER_INFO = ("HEADER", "TITLE", "COMPND", "SOURCE")
            pdb_path = f"{pdb_or_tempdir}/{sample['pdb_id']}.pdb"
            # pdb_path = f"{out_dir.rsplit('/',1)[0]}/download/{sample['pdb_id']}.pdb"

            header = list()
            with open(pdb_path, "r") as f:
                for line in f:
                    if line.startswith(HEADER_INFO):
                        header.append(line.strip())
                    else:
                        break

            strucio.save_structure(outfile, atom_array)

            with open(outfile, "r+") as f:
                content = f.read()
                f.seek(0, 0)
                print(*header, sep="\n", file=f)
                f.write(content)

        except Exception as E:
            log.error(
                f"PDB {sample['pdb_id']} {i+1}/{len(dataset)}: Unable to write predictions PDB: {E}"
            )
            traceback.print_exc()


class Clean_Chain(Select):
    def __init__(self, score, chain=None):
        self.score = score
        self.chain = chain
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

    def accept_chain(self, chain):
        return self.chain is None or chain == self.chain

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
    HEADER_INFO = ("HEADER", "TITLE", "COMPND", "SOURCE")

    p = PDBParser(PERMISSIVE=True)
    structure = p.get_structure(pdb_name, pdb_path)

    header = list()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(HEADER_INFO):
                header.append(line.strip())
            else:
                break

    chains = structure.get_chains()

    for chain in chains:
        pdb_out = f"{out_prefix}_{chain.get_id()}.pdb"
        io_w_no_h = PDBIO()
        io_w_no_h.set_structure(structure)
        with open(pdb_out, "w") as f:
            print(*header, sep="\n", file=f)
            io_w_no_h.save(f, Clean_Chain(score, chain))


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

    for i, prot_id in enumerate(pdb_list):

        if os.path.exists(f"{out_dir}/{prot_id}.pdb"):
            log.info(
                f"PDB {i+1} / {len(pdb_list)} ({prot_id}) already present: {out_dir}/{prot_id}.pdb"
            )
            continue

        else:
            log.info(f"Fetching {i+1}/{len(pdb_list)}: {prot_id}")

        if args.struc_type == "alphafold":
            URL = f"https://alphafold.ebi.ac.uk/files/AF-{prot_id}-F1-model_v4.pdb"
            score = None
        elif args.struc_type == "solved":
            URL = f"https://files.rcsb.org/download/{prot_id}.pdb"
            score = 100
        else:
            log.error(f"Structure ID was of unknown type {args.struc_type}")
            sys.exit(0)

        response = requests.get(URL)
        if response.status_code == 200:
            with open(f"{out_dir}/temp", "wb") as f:
                f.write(response.content)
        elif response.status_code == 404:
            log.error(f"File with the ID {prot_id} could not be found (url: {URL}).")
            log.error("Maybe you selected the wrong ID type or misspelled the ID.")
            log.error("Note that pdb files may not exist in RCSB for large structures - sorry for the inconvenience.")
            sys.exit(0)
        elif response.status_code in (408, 504):
            log.error(
                f"Request timed out with error code {response.status_code} (url: {URL})."
            )
            log.error(
                """Try to download the structure(s) locally from the given database and upload as pdb or zip.
                Bulk download script: https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script
                """
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

    if args.list_file and not args.struc_type:
        log.error(f"Must provide struc_type (solved or alphafold) with list_file")
        sys.exit()

    if args.pdb_or_zip_file and args.struc_type not in ["solved", "alphafold"]:
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
        print(args)
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

    # Check XGBoost models present
    models = glob.glob(f"{args.models_dir}/XGB_*_of_*.json")
    if len(models) != 100:
        log.error(f"Only found {len(models)}/100 models in {args.models_dir}")
        log.error(
            f"Did you download/unzip the model JSON files (e.g. XGB_1_of_100.json)?"
        )
        sys.exit(0)


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

    pass

    # Get basenames of input PDBs and output PDB/CSV files
    in_pdb_dict = get_directory_basename_dict(in_dir, "*.pdb")
    out_dict = get_directory_basename_dict(out_dir, "*.[pdb|csv]*")

    # Remove _discotope3 extension before comparison
    out_dict = {re.sub(r"_[A-Za-z]_discotope3$", "", k): v for k, v in out_dict.items()}

    # Log which input files are not found in output
    # missing_pdbs = in_pdb_dict.keys() - out_dict.keys()
    # if len(missing_pdbs) > 0:
    #    log.error(f"INFO: Failed processing PDBs: {', '.join(list(missing_pdbs))}")


def zip_folder_timeout(in_dir, out_dir) -> str:
    """Zips in_dir, writes to out_dir, returns zip file"""

    timestamp = time.strftime("%Y%m%d%H%M")
    file_name = f"discotope3_{timestamp}.zip"
    zip_path = f"{out_dir}/{file_name}"
    bashCommand = f"zip -j {zip_path} {in_dir}/*.pdb {in_dir}/*.csv || exit"

    try:
        output = subprocess.run(
            bashCommand, timeout=20, capture_output=True, shell=True
        )
        log.info(output.stdout.decode())
        return file_name
    except subprocess.TimeoutExpired:
        log.error("Error: zip compression timed out")
        sys.exit(0)


def main(args):
    """Main function"""

    # Error messages if invalid input
    check_valid_input(args)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as pdb_or_tempdir:

        pdb_or_tempdir = f"{args.out_dir}/download"
        os.makedirs(pdb_or_tempdir, exist_ok=True)

        # 1. Download PDBs from RCSB or AlphaFoldDB
        if args.list_file:
            log.info(f"Fetching PDBs")
            fetch_and_process_from_list_file(args.list_file, pdb_or_tempdir)

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
            pdb_or_tempdir,
            structure_type=args.struc_type,
            check_existing=args.check_existing,
            save_embeddings=args.save_embeddings,
            verbose=args.verbose,
        )
        if len(dataset) == 0:
            log.error("Error: No files in dataset.")
            sys.exit(0)

        # Predict and save
        log.info(f"Loading XGBoost ensemble")
        models = load_models(args.models_dir, num_models=100)  # MH

        log.info(f"Writing prediction .csv and .pdb files")
        write_model_prediction_csvs_pdbs(
            models,
            dataset,
            pdb_or_tempdir,
            out_dir=f"{args.out_dir}/output",
            verbose=args.verbose,
        )

        # Zip output folder
        log.info(f"Compressing ZIP file")
        out_zip = zip_folder_timeout(
            in_dir=f"{args.out_dir}/output", out_dir=f"{args.out_dir}/output"
        )

        # Check which files failed
        check_missing_pdb_csv_files(pdb_or_tempdir, f"{args.out_dir}/output")
        log.info(f"Done!")

        if args.web_server_mode:
            # HTML printing
            web_prefix = "/".join(f"{args.out_dir}/output".rsplit("/", 5)[1:])
            out_zip = f"{web_prefix}/{out_zip}"

            examples = """<script type="text/javascript">const examples = ["""
            structures = """<script type="text/javascript">const structures = ["""

            print("<h2>Output download</h2>")
            print(
                f'<a href="/{out_zip}"><p>Download DiscoTope-3.0 prediction results as zip</p></a>'
            )

            print(
                """<div class="wrap-collabsible">
                <input id="collapsible" class="toggle" type="checkbox">
                <label for="collapsible" class="lbl-toggle">Individual result downloads</label>
                <div class="collapsible-content">
                <div class="content-inner">
                """
            )

            for i, sample in enumerate(dataset):
                out_pdb = f"{web_prefix}/{sample['pdb_id']}_discotope3.pdb"
                out_csv = f"{web_prefix}/{sample['pdb_id']}_discotope3.csv"

                examples += "{"
                examples += f"id:'{sample['pdb_id']}',url:'https://services.healthtech.dtu.dk/{out_pdb}',info:'Structure {i+1}'"
                examples += "},"
                structures += "`"
                with open(
                    f"{args.out_dir}/output/{sample['pdb_id']}_discotope3.pdb", "r"
                ) as f:
                    structures += f.read()
                structures += "`,"

                style = ' style="margin-top:1em;"' if i > 0 else ""
                print(f"<h3{style}>{sample['pdb_id']}</h3>")
                print(
                    f'<a href="/{out_pdb}"><span>Download PDB w/ DiscoTope-3.0 prediction scores</span></a>'
                )
                print(f'<a href="/{out_csv}"><span>Download CSV</span></a> <br>')

            print("</div></div></div>")
            examples += "];</script>"
            structures += "];</script>"
            print(examples)
            print(structures)


if __name__ == "__main__":

    args = cmdline_args()
    os.makedirs(f"{args.out_dir}/output", exist_ok=True)
    logging.basicConfig(
        filename=f"{args.out_dir}/output/dt3.log",
        encoding="utf-8",
        level=logging.INFO,
        format="[{asctime}] {message}",
        style="{",
    )
    log = logging.getLogger(__name__)
    log.info("Predicting PDBs using Discotope-3.0")

    # Print logging/errors to console if web server mode (viewable HTML)
    if args.web_server_mode:
        logging.getLogger().addHandler(logging.StreamHandler())

    # Verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        main(args)
    except Exception as E:
        log.exception(
            f"Prediction encountered an unexpected error. This is likely a bug in the server software: {E}"
        )
