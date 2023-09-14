# Discotope 3.0 - Predict B-cell epitope propensity from structure
# https://github.com/Magnushhoie/discotope3_web

MAX_FILES_INT = 100
MAX_FILE_SIZE_MB = 50

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
import pickle
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
from main import (get_basename_no_ext,
                               get_directory_basename_dict, normalize_scores,
                               predict_using_models, read_list_file,
                               set_struc_res_bfactor, true_if_zip,
                               save_clean_pdb_single_chains,
                               predict_and_save,
                               load_models,
                               load_gam_model,
                               save_clean_pdb_single_chains,
                               fetch_pdbs_extract_single_chains,
                               )

def cmdline_args():
    # Make parser object
    usage = rf"""
Options:
    1) Single PDB file, ZIP or list (--list_or_pdb_or_zip_file <file>)
    3) Directory with PDB files (--pdb_dir <folder>)
 
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
        "--list_or_pdb_or_zip_file",
        dest="list_or_pdb_or_zip_file",
        help="File with PDB or Uniprot IDs, fetched from RCSB/AlphaFolddb (1) or single PDB input file (2) or compressed zip file with multiple PDBs (3)",
        type=lambda x: is_valid_path(p, x),
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
        "--calibrated_score_epi_threshold",
        type=float,
        help="Calibrated-score threshold for epitopes [low 0.40, moderate (0.90), higher 1.50]",
        default=0.90,
    )

    p.add_argument(
        "--no_calibrated_normalization",
        action="store_true",
        default=False,
        help="Skip Calibrated-normalization of PDBs",
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
        "--multichain_mode",
        action="store_true",
        default=False,
        help="Predicts entire complexes, unsupported and untested",
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

    # pdb_or_zip_file
    # list_file

    # Check input arguments
    if not (args.list_or_pdb_or_zip_file):
        log.error(
            f"""Please choose one of:
        1) PDB file
        2) Zip file with PDBs
        4) File with PDB ids on each line
        """
        )
        sys.exit(0)

    if args.list_or_pdb_or_zip_file and not args.struc_type:
        log.error(f"Must provide struc_type (solved or alphafold)")
        sys.exit(0)

    if args.list_or_pdb_or_zip_file and args.struc_type not in ["solved", "alphafold"]:
        log.error(
            f"--struc_type flag invalid, must be solved or alphafold. Found {args.struc_type}"
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

    size_mb = os.stat(args.list_or_pdb_or_zip_file).st_size / (1024 * 1024)
    if size_mb > MAX_FILES_INT:
        log.error(f"Max file-size {MAX_FILE_SIZE_MB} MB, found {round(size_mb)} MB")
        sys.exit(0)

    if true_if_list(args.list_or_pdb_or_zip_file):
        return True  # IS LIST FILE

    # Check ZIP max-size, number of files
    if true_if_zip(args.list_or_pdb_or_zip_file):
        with closing(ZipFile(args.list_or_pdb_or_zip_file)) as archive:
            file_count = len(archive.infolist())
            file_names = archive.namelist()

        # Check number of files in zip
        if file_count > MAX_FILES_INT:
            log.error(f"Max number of files {file_count}, found {file_count}")
            sys.exit(0)

        # Check filenames end in .pdb
        name = file_names[0]
        if os.path.splitext(name)[-1] != ".pdb":
            log.error(f"Ensure all ZIP content file-names end in .pdb, found {name}")
            sys.exit(0)
        return  # IS NOT LIST FILE


def report_pdb_input_outputs(pdb_list, in_dir, out_dir) -> None:
    """Reports missing CSV and PDB file in out_dir, per PDB file in in_dir"""

    in_pdbs = glob.glob(f"{in_dir}/*.pdb")
    out_pdbs = glob.glob(f"{out_dir}/*.pdb")

    in_dict = {Path(pdb).stem: pdb for pdb in in_pdbs}
    out_dict = {re.sub(r"_discotope3$", "", Path(pdb).stem): pdb for pdb in out_pdbs}

    # Report number of input vs outputs
    log.info(
        f"Predicted {len(out_pdbs)} / {len(in_pdbs)} single PDB chain(s) from {len(pdb_list)} input PDBs, saved to {out_dir}"
    )

    missing_pdbs = in_dict.keys() - out_dict.keys()
    if len(missing_pdbs) >= 1:
        log.info(
            f"Note: Excluded predicting {len(missing_pdbs)} PDB chain(s) (see log file):\n{', '.join(missing_pdbs)}"
        )


def print_HTML_output_webpage(dataset, out_dir) -> None:
    """Hardcoded HTML output for download links to results"""

    # HTML printing
    examples = """<script type="text/javascript">const examples = ["""
    structures = """<script type="text/javascript">const structures = ["""

    OUTPUT_HTML = ""

    for i, sample in enumerate(dataset):
        examples += "{"
        examples += f"id:'{sample['pdb_id']}',url:'{sample['pdb_id']}_discotope3.pdb',info:'Structure {i+1}'"
        examples += "},"
        structures += "`"
        with open(f"{out_dir}/{sample['pdb_id']}_discotope3.pdb", "r") as f:
            structures += f.read()
        structures += "`,"

    examples += "];</script>"
    structures += "];</script>"
    OUTPUT_HTML += examples
    OUTPUT_HTML += structures

    with open("/output.html", "r") as f:
        output_html = f.read()

    output_html_w_data = output_html.replace("INSERT_OUTPUT_HERE", OUTPUT_HTML)
    with open("/output.html", "w") as f:
        f.write(output_html_w_data)


def true_if_list(infile):
    """Returns True if file header bits are zip file"""
    with open(infile, "rb") as f:
        first_line = f.readline().strip()
    PDB_REGEX = rb"^[0-9][A-Za-z0-9]{3}"
    UNIPROT_REGEX = (
        rb"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
    )
    return (re.search(PDB_REGEX, first_line) is not None) or (
        re.search(UNIPROT_REGEX, first_line) is not None
    )


def main(args):
    """Main function"""

    # Log if multichain mode is set
    if args.multichain_mode:
        log.info(f"Multi-chain mode set, will predict PDBs as complexes")
    else:
        log.info(f"Single-chain mode set, will predict PDBs as single chains")

    # Error messages if invalid input
    is_list_file = check_valid_input(args)

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
    if is_list_file:
        log.debug(f"Fetching PDBs")

        pdb_list = read_list_file(args.list_or_pdb_or_zip_file)
        log.info(
            f"Fetching {len(pdb_list)} PDBs from {'RCSB' if args.struc_type == 'solved' else 'AlphaFoldDB'}, extracting single chains to {input_chains_dir}"
        )
        fetch_pdbs_extract_single_chains(pdb_list, input_chains_dir)

    # Check whether input file is a compressed ZIP file or single PDB
    else:
        # 2. If ZIP, unzip and extract single chains
        if true_if_zip(args.list_or_pdb_or_zip_file):
            log.debug(f"Unzipping PDBs")

            # Temporary directory for zip output, delete after extract/saving single chains
            with tempfile.TemporaryDirectory() as tempdir:
                zf = ZipFile(args.list_or_pdb_or_zip_file)
                zf.extractall(tempdir)

                pdb_list = glob.glob(f"{tempdir}/*.pdb")
                log.info(
                    f"Extracted {len(pdb_list)} PDBs from ZIP, extracting single chains to {input_chains_dir}"
                )
                for f in pdb_list:
                    pdb_name = get_basename_no_ext(f)
                    save_clean_pdb_single_chains(
                        f,
                        pdb_name,
                        bscore,
                        input_chains_dir,
                        save_full_complex=args.multichain_mode,
                    )

        # 3. If single PDB, copy to tempdir
        else:
            f = args.list_or_pdb_or_zip_file
            pdb_name = get_basename_no_ext(f)
            pdb_list = [pdb_name]
            log.info(
                f"Single PDB file input ({pdb_name}), extracting single chains to {input_chains_dir}"
            )
            save_clean_pdb_single_chains(
                f,
                pdb_name,
                bscore,
                input_chains_dir,
                save_full_complex=args.multichain_mode,
            )

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

    # Load GAMs to normalize scores by length and surface area
    gam_len_to_mean = load_gam_model("/discotope3_web/models/gam_len_to_mean.pkl")
    gam_surface_to_std = load_gam_model(
        "/discotope3_web/models/gam_surface_to_std.pkl"
    )

    # Predict and save
    predict_and_save(
        models,
        dataset,
        input_chains_dir,
        out_dir=out_dir,
        gam_len_to_mean=gam_len_to_mean,
        gam_surface_to_std=gam_surface_to_std,
        calibrated_score_epi_threshold=args.calibrated_score_epi_threshold,
        no_calibrated_normalization=args.no_calibrated_normalization,
        verbose=args.verbose,
    )

    # Print if any PDB input / output summary, and any missing PDBs
    report_pdb_input_outputs(pdb_list, input_chains_dir, out_dir)

    # If web server mode, prepare downloadable zip and results HTML page
    if args.web_server_mode:
        # Prints per single chain result download links and HTML output
        # Skips over any PDBs that are missing results
        log.debug(f"Printing HTML output")
        print_HTML_output_webpage(dataset, out_dir)


if __name__ == "__main__":
    args = cmdline_args()
    os.makedirs(f"{args.out_dir}/output/", exist_ok=True)

    # Load ESM-IF1 from LFS
    ESM_MODEL_DIR = "/root/.cache/torch/hub/checkpoints/"
    ESM_MODEL_FILE = "esm_if1_gvp4_t16_142M_UR50.pt"
    os.makedirs(ESM_MODEL_DIR, exist_ok=True)
    os.symlink(
        f"{args.models_dir}/{ESM_MODEL_FILE}", f"{ESM_MODEL_DIR}/{ESM_MODEL_FILE}"
    )

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
