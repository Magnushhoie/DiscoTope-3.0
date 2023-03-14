# Discotope 3.0 - Predict B-cell epitope propensity from structure
# https://github.com/Magnushhoie/discotope3_web

MAX_FILES = 50
MAX_FILE_SIZE_MB = 30

import logging
import os
import sys
# Ignore Biopython deprecation warnings
import warnings

from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

import copy
import glob
import os
import re
import subprocess
import tempfile
import time
from argparse import ArgumentParser, RawTextHelpFormatter
from contextlib import closing
from pathlib import Path
from typing import List
from zipfile import ZipFile

import biotite
import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from Bio.PDB import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser

from make_dataset import Discotope_Dataset_web


def cmdline_args():
    # Make parser object
    usage = rf"""
Options:
    1) Single PDB file (--pdb_or_zip_file <file>)
    2) Zip file containing PDBs (--pdb_or_zip_file <file>)
    3) Directory with PDB files (--pdb_dir <folder>)
    4) File with PDB ids on each line (--list_file <file>)

# Predict on example PDBs in data folder
python src/predict_webserver.py \
--pdb_dir data/example_pdbs_solved \
--struc_type solved \
--out_dir output/example_pdbs_solved

# Fetch PDBs from list file from AlphaFoldDB
python src/predict_webserver.py \
--list_file data/pdb_list_af2.txt \
--struc_type alphafold \
--out_dir output/pdb_list_af2
 
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
        action="store_true",
        default=False,
        help="Flag for printing HTML output",
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
        required=True,  # Only needed for file input, not list
        default="solved",
        help="Structure type from file (solved | alphafold)",
    )

    p.add_argument(
        "--pdb_dir",
        help="Directory with AF2 PDBs",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--out_dir",
        default="output/",
        required=True,
        help="Job output directory",
    )

    p.add_argument(
        "--models_dir",
        default="models/",
        help="Path for .json files containing trained XGBoost ensemble",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--check_existing_embeddings",
        default=False,
        help="Check for existing embeddings to load in pdb_dir",
    )

    p.add_argument(
        "--cpu_only",
        action="store_true",
        default=False,
        help="Use CPU even if GPU is available (default uses GPU if available)",
    )

    p.add_argument(
        "--max_gpu_pdb_length",
        type=int,
        help="Maximum PDB length to embed on GPU (1000), otherwise CPU",
        default=1000,
    )

    p.add_argument(
        "--save_embeddings",
        default=False,
        help="Save embeddings to pdb_dir",
    )

    p.add_argument("-v", "--verbose", type=int, default=1, help="Verbose logging")

    # Print help if no arguments
    if len(sys.argv) == 1:
        p.print_help(sys.stderr)
        sys.exit(1)

    return p.parse_args()


def check_valid_input(args):
    """Checks for valid arguments"""

    # Check input arguments
    if not (args.pdb_or_zip_file or args.pdb_dir or args.list_file):
        if args.web_server_mode:
            log.error(
                f"""Please provide one of:
            a) Upload PDB file or ZIP file containing PDB files
            b) A list of protein structure IDs
            """
            )
        else:
            log.error(
                f"""Please choose one of:
            1) PDB file (--pdb_or_zip_file)
            2) Zip file with PDBs (--pdb_or_zip_file)
            3) PDB directory (--pdb_dir)
            4) File with PDB ids on each line (--list_file)
            """
            )
        sys.exit(1)

    if args.list_file and not os.path.exists(args.list_file):
        log.error(f"Error: List file {args.list_file} does not exist!")

    if args.list_file and not args.struc_type:
        log.error(
            f"Error: Must provide struc_type (solved or alphafold) with list_file"
        )
        sys.exit()

    if args.pdb_or_zip_file and args.struc_type not in ["solved", "alphafold"]:
        log.error(
            f"Error: --struc_type {args.struc_type} is invalid, must be either 'solved' or 'alphafold'"
        )
        sys.exit(1)

    if (
        (args.pdb_dir and args.list_file)
        or (args.pdb_dir and args.pdb_or_zip_file)
        or (args.list_file and args.pdb_or_zip_file)
    ):
        log.error(
            f"Error: Please choose only one input option: pdb_dir, list_file or pdb_or_zip_file"
        )
        print(args)
        sys.exit(1)

    # Check ZIP max-size, number of files
    if args.pdb_or_zip_file:
        size_mb = os.stat(args.pdb_or_zip_file).st_size / (1024 * 1024)
        if args.web_server_mode and size_mb > MAX_FILES:
            log.error(
                f"Error: Max file-size {MAX_FILE_SIZE_MB} MB, found {round(size_mb)} MB"
            )
            sys.exit(1)

        if true_if_zip(args.pdb_or_zip_file):
            with closing(ZipFile(args.pdb_or_zip_file)) as archive:
                file_count = len(archive.infolist())
                file_names = archive.namelist()

            # Check number of files in zip
            if file_count > MAX_FILES:
                log.error(
                    f"Error: Max number of files {file_count}, found {file_count}"
                )
                sys.exit(1)

            # Check filenames end in .pdb
            name = file_names[0]
            if os.path.splitext(name)[-1] != ".pdb":
                log.error(
                    f"Error: Ensure all ZIP content file-names end in .pdb, found {name}"
                )
                sys.exit(1)

    # Check XGBoost models present
    models = glob.glob(f"{args.models_dir}/XGB_*_of_*.json")
    if len(models) != 100:
        log.error(
            f"Error: Found {len(models)}/100 XGBoost model JSON files in {args.models_dir}"
        )
        log.error(
            f"Did you unzip the models.zip file? models/ should 100 XGBoost model files, with names XGB_n_of_100.json"
        )
        sys.exit(1)


def true_if_zip(infile):
    """Returns True if file header bits are zip file"""
    with open(infile, "rb") as fb:
        header_bits = fb.read(4)
    return header_bits == b"PK\x03\x04"


def load_models(
    models_dir: str,
    num_models: int = 100,
    verbose: int = 1,
) -> List["xgb.XGBClassifier"]:
    """Loads saved XGBoostClassifier files containing model weights, returns list of XGBoost models"""
    import xgboost as xgb

    # Search for model files
    model_files = list(Path(models_dir).glob(f"XGB_*_of_*.json"))

    if len(model_files) == 0:
        log.error(f"Error: no files found in {models_dir}.")
        sys.exit(1)

    # Initialize new XGBoostClassifier and load model weights
    log.debug(
        f"Loading {num_models} / {len(model_files)} XGBoost models from {models_dir}"
    )

    models = []
    for fp in model_files[:num_models]:
        m = xgb.XGBClassifier()
        m.load_model(str(fp))
        models.append(m)

    return models


def predict_using_models(
    models: List["xgb.XGBClassifier"],
    X: "np.array",
) -> "np.array":
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


def set_struc_res_bfactor(atom_array, res_values):
    """Set per-residue B-factor of atom_array to res_values (DiscoTope-3.0 predictions)"""

    # Get relative indices starting from 1. Copy to avoid modifying original
    atom_array_renum = biotite.structure.renumber_res_ids(
        copy.deepcopy(atom_array), start=1
    )
    atom_array.b_factor = res_values[atom_array_renum.res_id - 1]

    return atom_array


def predict_and_save(models, dataset, pdb_dir, out_dir, verbose: int = 0) -> None:
    """Predicts and saves CSV/PDBs with DiscoTope-3.0 scores"""

    log.debug(f"Predicting PDBs ...")

    # Speed up predictions by predicting entire dataset, exclude missing PDBs
    X_all = np.concatenate(
        [
            dataset[i]["X_arr"]
            for i in range(len(dataset))
            if dataset[i]["X_arr"] is not False
        ]
    )
    y_all = predict_using_models(models, X_all)

    # Put in predictions for each PDB dataframe
    df_all = pd.concat(
        [
            dataset[i]["df_stats"]
            for i in range(len(dataset))
            if dataset[i]["X_arr"] is not False
        ]
    )
    df_all.insert(3, "DiscoTope-3.0_score", y_all)

    # Round numerical columns to 5 digits for nicer CSV output
    num_cols = ["DiscoTope-3.0_score", "rsa"]
    df_all[num_cols] = df_all[num_cols].applymap(lambda x: "{:.5f}".format(x))

    # Keep track of structures for later
    strucs_all = [
        dataset[i]["PDB_biotite"]
        for i in range(len(dataset))
        if dataset[i]["X_arr"] is not False
    ]

    # PDB lengths used to infer which PDBs the predictions belong to
    X_lens = [
        len(dataset[i]["X_arr"])
        for i in range(len(dataset))
        if dataset[i]["X_arr"] is not False
    ]
    pdbids_all = [
        dataset[i]["pdb_id"]
        for i in range(len(dataset))
        if dataset[i]["X_arr"] is not False
    ]

    if len(X_all) != sum(X_lens):
        log.error(
            f"Software bug: PDB feature array length {len(X_all)} does not match summed PDB lengths {sum(X_lens)}"
        )
        sys.exit(1)

    if len(X_all) != len(df_all):
        log.error(
            f"Software bug: PDB feature array length {len(df_all)} does not match merged PDBs dataframe length {len(df_all)}"
        )
        sys.exit(1)

    # Fetch pre-computed predictions by PDB lengths, save to CSV/PDB
    start = 0
    for _pdb, L, struc in zip(pdbids_all, X_lens, strucs_all):
        log.debug(
            f"Saving predictions for {_pdb} CSV/PDB to {out_dir}/{_pdb}_discotope3.csv"
        )

        end = start + L
        df = df_all.iloc[start:end]
        start = end

        # Save CSV
        outfile = f"{out_dir}/{_pdb}_discotope3.csv"
        df.to_csv(outfile, index=False)

        # Save PDB, after adding prediction scores
        struc_pred = set_struc_res_bfactor(
            struc, df["DiscoTope-3.0_score"].values.astype(float) * 100
        )
        outfile = f"{out_dir}/{_pdb}_discotope3.pdb"
        strucio.save_structure(outfile, struc_pred)


def save_clean_pdb_single_chains(pdb_path, pdb_name, bscore, outdir):
    class Clean_Chain(Select):
        def __init__(self, score, chain=None):
            self.bscore = bscore
            self.chain = chain
            if bscore is None:
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
                atom.set_bfactor(self.bscore)
            else:
                self.letter = atom.get_full_id()[3][2]
                if atom.get_full_id()[3][2] not in (self.prev_letter, " "):
                    log.debug(
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

                atom.set_bfactor(self.bscore[res_id - self.init_resid])
            return True

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
        pdb_out = f"{outdir}/{pdb_name}_{chain.get_id()}.pdb"
        io_w_no_h = PDBIO()
        io_w_no_h.set_structure(structure)
        with open(pdb_out, "w") as f:
            print(*header, sep="\n", file=f)
            io_w_no_h.save(f, Clean_Chain(bscore, chain))


def fetch_pdbs_extract_single_chains(pdb_list, out_dir) -> None:
    """Fetch and process PDB chains/UniProt entries from list input"""

    if len(pdb_list) == 0:
        log.error("Error: No IDs found in PDB list.")
        sys.exit(1)

    elif args.web_server_mode and len(pdb_list) > MAX_FILES:
        log.error(
            f"Error: A maximum of {MAX_FILES} PDB IDs can be processed at one time. ({len(pdb_list)} IDs found)."
        )
        sys.exit(1)

    for i, prot_id in enumerate(pdb_list):
        if os.path.exists(f"{out_dir}/{prot_id}.pdb"):
            log.debug(
                f"PDB {i+1} / {len(pdb_list)} ({prot_id}) already present: {out_dir}/{prot_id}.pdb"
            )
            continue

        else:
            log.debug(f"Fetching {i+1}/{len(pdb_list)}: {prot_id}")

        if args.struc_type == "alphafold":
            URL = f"https://alphafold.ebi.ac.uk/files/AF-{prot_id}-F1-model_v4.pdb"
            bscore = None
        elif args.struc_type == "solved":
            URL = f"https://files.rcsb.org/download/{prot_id}.pdb"
            bscore = 100
        else:
            log.error(f"Error: Structure ID is of unknown type {args.struc_type}")
            sys.exit(1)

        response = requests.get(URL)
        if response.status_code == 200:
            with open(f"{out_dir}/temp", "wb") as f:
                f.write(response.content)

        elif response.status_code == 404:
            log.error(
                f"Error: File with the ID {prot_id} could not be found (url: {URL})."
            )
            log.error("Maybe you selected the wrong ID type or misspelled the ID?")
            log.error("Note that PDB files may not exist in RCSB for large structures")
            sys.exit(1)

        elif response.status_code in (408, 504):
            log.error(
                f"Error: Request timed out with error code {response.status_code} (url: {URL})."
            )
            log.error(
                """Try to download the structure(s) locally from the given database and upload individually or as compressed zip with PDBs.
                Bulk download script: https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script
                """
            )
            sys.exit(1)

        else:
            log.error(
                f"Error: Received status code {response.status_code}, when trying to fetch file from {URL}"
            )
            sys.exit(1)

        save_clean_pdb_single_chains(
            pdb_path=f"{out_dir}/temp", pdb_name=prot_id, bscore=bscore, outdir=out_dir
        )


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


def report_pdb_input_outputs(pdb_list, in_dir, out_dir) -> None:
    """Reports missing CSV and PDB file in out_dir, per PDB file in in_dir"""

    in_pdbs = glob.glob(f"{in_dir}/*.pdb")
    out_pdbs = glob.glob(f"{out_dir}/*.pdb")

    in_dict = {Path(pdb).stem: pdb for pdb in in_pdbs}
    out_dict = {re.sub(r"_discotope3$", "", Path(pdb).stem): pdb for pdb in out_pdbs}

    # Report number of input vs outputs
    log.info(
        f"Predicted {len(out_pdbs)} / {len(in_pdbs)} single PDB chain(s) from {len(pdb_list)} input PDBs, saved to {args.out_dir}/output"
    )

    missing_pdbs = in_dict.keys() - out_dict.keys()
    if len(missing_pdbs) >= 1:
        log.info(
            f"Note: Excluded predicting {len(missing_pdbs)} PDB chain(s) (see log file):\n{', '.join(missing_pdbs)}"
        )


def zip_folder_timeout(in_dir, out_dir, timeout_seconds=120) -> str:
    """Zips in_dir, writes to out_dir, returns zip file"""

    timestamp = time.strftime("%Y%m%d%H%M")
    file_name = f"discotope3_{timestamp}.zip"
    zip_path = f"{out_dir}/{file_name}"
    bashCommand = (
        f"zip -j {zip_path} {in_dir}/log.txt {in_dir}/*.pdb {in_dir}/*.csv || exit"
    )

    try:
        output = subprocess.run(
            bashCommand, timeout=timeout_seconds, capture_output=True, shell=True
        )
        log.debug(output.stdout.decode())
        return file_name

    except subprocess.TimeoutExpired:
        log.error("Error: zip compression timed out")
        sys.exit(1)


def print_HTML_output_webpage(dataset, out_dir, out_zip) -> None:
    """Hardcoded HTML output for download links to results"""

    # Variables for HTML printing
    web_prefix = "/".join(f"{out_dir}".rsplit("/", 5)[1:])
    out_zip = f"{web_prefix}/{out_zip}"

    examples = """<script type="text/javascript">const examples = ["""
    structures = """<script type="text/javascript">const structures = ["""

    # Header
    print("</span></div></div></div>")
    print("<h2>Output download</h2>")
    print(
        f'<a href="/{out_zip}"><p>Download DiscoTope-3.0 prediction results as zip</p></a>'
    )

    # Collapsible box for individual results
    print(
        """<div class="wrap-collabsible">
        <input id="collapsible-1" class="toggle" type="checkbox">
        <label for="collapsible-1" class="lbl-toggle">Individual result downloads</label>
        <div class="collapsible-content">
        <div class="content-inner">
        """
    )

    # Add individual results
    for i, sample in enumerate(dataset):
        # Only include present PDBs
        if not type(sample["X_arr"]):
            continue

        else:
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

    # End of collapsible box
    print("</div></div></div>")
    examples += "];</script>"
    structures += "];</script>"
    print(examples)
    print(structures)


def read_list_file(list_file) -> List[str]:
    """Reads list file, returns list of PDB IDs"""

    with open(list_file, "r") as f:
        pdb_list = [line.strip() for line in f.readlines()]

    return pdb_list


def main(args):
    """Main function"""

    # Directory for input single chains (extracted from input PDBs) and output CSV/PDB results
    input_chains_dir = f"{args.out_dir}/input_chains"
    out_dir = f"{args.out_dir}/output"

    os.makedirs(input_chains_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Set B-factor / pLDDT column to 100 for solved structures, leave as is for AF2 structures
    if args.struc_type == "alphafold":
        bscore = 100
    else:
        bscore = None

    # Prepare input PDBs by extracting single chains to input_chains_dir
    # Nb: After check_valid_inputs, only one of [pdb_or_zip_file, list_file, pdb_dir] is set

    # 1. Download PDBs from RCSB or AlphaFoldDB
    if args.list_file:
        log.debug(f"Fetching PDBs")

        pdb_list = read_list_file(args.list_file)
        log.info(
            f"Fetching {len(pdb_list)} PDBs from {'RCSB' if args.struc_type == 'solved' else 'AlphaFoldDB'}, extracting single chains to {input_chains_dir}"
        )
        fetch_pdbs_extract_single_chains(pdb_list, input_chains_dir)

    # Check whether input file is a compressed ZIP file or single PDB
    if args.pdb_or_zip_file:
        # 2. If ZIP, unzip and extract single chains
        if true_if_zip(args.pdb_or_zip_file):
            log.debug(f"Unzipping PDBs")

            # Temporary directory for zip output, delete after extract/saving single chains
            with tempfile.TemporaryDirectory() as tempdir:
                zf = ZipFile(args.pdb_or_zip_file)
                zf.extractall(tempdir)

                pdb_list = glob.glob(f"{tempdir}/*.pdb")
                log.info(
                    f"Extracted {len(pdb_list)} PDBs from ZIP, extracting single chains to {input_chains_dir}"
                )
                for f in pdb_list:
                    pdb_name = get_basename_no_ext(f)
                    save_clean_pdb_single_chains(f, pdb_name, bscore, input_chains_dir)

        # 3. If single PDB, copy to tempdir
        else:
            f = args.pdb_or_zip_file
            pdb_name = get_basename_no_ext(f)
            pdb_list = [pdb_name]
            log.info(
                f"Single PDB file input ({pdb_name}), extracting single chains to {input_chains_dir}"
            )
            save_clean_pdb_single_chains(f, pdb_name, bscore, input_chains_dir)

    # 4. Load from PDB folder
    if args.pdb_dir:
        pdb_list = glob.glob(f"{args.pdb_dir}/*.pdb")
        log.info(
            f"Found {len(pdb_list)} PDBs in {args.pdb_dir}, extracting single chains to {input_chains_dir}"
        )
        for f in pdb_list:
            pdb_name = get_basename_no_ext(f)
            save_clean_pdb_single_chains(f, pdb_name, bscore, input_chains_dir)

    # Summary statistics
    chain_list = glob.glob(f"{input_chains_dir}/*.pdb")
    log.info(f"Found {len(chain_list)} extracted single chains in {input_chains_dir}")

    # Embed end process PDB features
    log.debug(f"Pre-processing PDBs")
    dataset = Discotope_Dataset_web(
        input_chains_dir,
        structure_type=args.struc_type,
        check_existing_embeddings=args.check_existing_embeddings,
        save_embeddings=args.save_embeddings,
        cpu_only=args.cpu_only,
        max_gpu_pdb_length=args.max_gpu_pdb_length,
        verbose=args.verbose,
    )
    if len(dataset) == 0:
        log.error("Error: No PDB files were valid. Please check input PDBs.")
        sys.exit(1)

    # Load pre-trained XGBoost models
    models = load_models(args.models_dir, num_models=100)

    # Predict and save
    predict_and_save(
        models,
        dataset,
        input_chains_dir,
        out_dir=out_dir,
        verbose=args.verbose,
    )

    # Print if any PDB input / output summary, and any missing PDBs
    report_pdb_input_outputs(pdb_list, input_chains_dir, out_dir)

    # If web server mode, prepare downloadable zip and results HTML page
    if args.web_server_mode:
        # Prepare downloadable zip, store in same output directory
        log.debug(f"Compressing ZIP file")
        out_zip = zip_folder_timeout(in_dir=out_dir, out_dir=out_dir)

        # Prints per single chain result download links and HTML output
        # Skips over any PDBs that are missing results
        log.debug(f"Printing HTML output")
        print_HTML_output_webpage(dataset, out_dir, out_zip)


if __name__ == "__main__":
    args = cmdline_args()
    os.makedirs(f"{args.out_dir}/output/", exist_ok=True)

    # Log to file and stdout
    # If verbose == 0, only errors are printed (default 1)
    log_path = os.path.abspath(f"{args.out_dir}/output/log.txt")
    logging.basicConfig(
        level=logging.ERROR,
        format="[{asctime}] {message}",
        style="{",
        handlers=[
            logging.FileHandler(filename=log_path, mode="w"),
            logging.StreamHandler(stream=sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)

    # Error messages if invalid input
    check_valid_input(args)

    # INFO prints total summary and errors (default)
    if args.web_server_mode or args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)

    # DEBUG prints every major step
    elif args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)

    # Exclude deprecation warnings (from Biopython, etc.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            log.debug("Predicting PDBs using Discotope-3.0")
            main(args)
            log.debug("Done!")

        except Exception as E:
            log.exception(
                f"Prediction encountered an unexpected error. This is likely a bug in the server software: {E}"
            )
