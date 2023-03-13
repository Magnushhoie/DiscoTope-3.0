# ROOT_Path
import os
import sys
from pathlib import Path

ROOT_PATH = str(Path(os.getcwd()))
sys.path.insert(0, "ROOT_PATH")

import logging

# Nb: assume logging is already configured in main script
# logging.basicConfig(level=logging.ERROR, format="[{asctime}] {message}", style="{")
log = logging.getLogger(__name__)

import copy
import glob
import re
import traceback
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import List

import biotite.structure
import numpy as np
import pandas as pd
import torch
# Use Sander values instead
from biotite.structure import filter_amino_acids, filter_backbone, get_chains
from biotite.structure.io import pdb, pdbx
from biotite.structure.residues import get_residues
from joblib import Parallel, delayed

import esm_util_custom

# Sander scale residue SASA values
# from Bio.Data.PDBData import residue_sasa_scales
sasa_max_dict = {
    "A": 106.0,
    "R": 248.0,
    "N": 157.0,
    "D": 163.0,
    "C": 135.0,
    "Q": 198.0,
    "E": 194.0,
    "G": 84.0,
    "H": 184.0,
    "I": 169.0,
    "L": 164.0,
    "K": 205.0,
    "M": 188.0,
    "F": 197.0,
    "P": 136.0,
    "S": 130.0,
    "T": 142.0,
    "W": 227.0,
    "Y": 222.0,
    "V": 142.0,
    "X": 169.55,  # Use mean for non-standard residues
}


def cmdline_args():
    # Make parser object
    usage = f"""
    # Make dataset with test set
    python src/data/make_dataset.py \
    --pdb_dir data/raw/af2_pdbs \
    --out_dir data/processed/test__1
    """
    p = ArgumentParser(
        description="Make dataset",
        formatter_class=RawTextHelpFormatter,
        usage=usage,
    )

    def is_valid_path(parser, arg):
        if not os.path.exists(arg):
            parser.error(f"Path {arg} does not exist!")
        else:
            return arg

    p.add_argument(
        "--struc_type",
        required=True,
        help="Type of PDB structure (solved | alphafold)",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Job output directory",
        metavar="FOLDER",
    )
    p.add_argument(
        "--pdb_dir",
        default=f"{ROOT_PATH}/raw/af2_pdbs",
        required=False,
        help="Directory with AF2 PDBs",
        metavar="FOLDER",
        type=lambda x: is_valid_path(p, x),
    )
    p.add_argument(
        "--models_path",
        default=f"{ROOT_PATH}/models/",
        required=False,
        help="Path for saving XGBoost models",
        metavar="FOLDER",
    )
    p.add_argument(
        "--max_gpu_pdb_length",
        type=int,
        help="Maximum PDB length to embed on GPU (1000), otherwise CPU",
        default=1000,
    )
    p.add_argument(
        "--skip_embeddings",
        type=bool,
        help="Skip ESM-IF1 embedding step (False)",
        default=False,
    )
    p.add_argument(
        "--overwrite_embeddings",
        type=bool,
        help="Recreate PDB ESM-IF1 embeddings even if existing",
        default=False,
    )
    p.add_argument(
        "-v", "--verbose", dest="verbose", default=0, type=int, help="Verbose logging"
    )

    return p.parse_args()


def load_IF1_tensors(
    pdb_files: List,
    check_existing_embeddings=True,
    save_embeddings=True,
    cpu_only=False,
    max_gpu_pdb_length: int = 1000,
    verbose: int = 0,
) -> List:
    """Generate or load ESM-IF1 embeddings for a list of PDB files"""

    import esm
    import esm.inverse_folding

    # Load ESM inverse folding model
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    # Lists containing final output for each PDB
    list_IF1_tensors = []
    list_structures = []
    list_sequences = []
    for i, pdb_path in enumerate(pdb_files):
        # Get PDB name
        _pdb = get_basename_no_ext(pdb_path)

        # Try to extract backbone, or skip
        try:
            structure, structure_full = load_structure_discotope(
                str(pdb_path), chain=None
            )
        except Exception as E:
            log.error(f"Note: Skipping {_pdb}, no valid amino-acid backbone found")
            log.debug(f"Error: {E}")
            list_IF1_tensors.append(False)
            list_structures.append(False)
            list_sequences.append(False)
            continue

        # Try to extract C, Ca, N atoms and sequence, or skip
        try:
            coords, seq = esm_util_custom.extract_coords_from_structure(structure)

        except Exception as E:
            log.error(
                f"Error: Unable to extract valid amino-acid sequence / backbone from {_pdb}"
            )
            log.debug(f"Error: {E}")
            list_IF1_tensors.append(False)
            list_structures.append(False)
            list_sequences.append(False)
            continue

        # Load IF1 tensor if already exists and flag is set (default always embed from scratch)
        embed_file = re.sub(r".pdb$", ".pt", pdb_path)
        if check_existing_embeddings and os.path.exists(embed_file):
            log.debug(
                f"{i+1} / {len(pdb_files)}: Loading existing embedding file for {_pdb}: {embed_file}"
            )
            rep = torch.load(embed_file)

        # Else, embed PDB from scratch
        else:
            log.debug(f"{i+1} / {len(pdb_files)}: Embedding {_pdb}")

            # Try to embed on CPU/GPU and save
            try:
                # Embed on GPU if available, unless sequence length > (1000) or cpu_only flag is set
                device = torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    and len(seq) < max_gpu_pdb_length
                    and not cpu_only
                    else "cpu"
                )

                rep = (
                    esm_util_custom.get_encoder_output(
                        model, alphabet, coords, seq, device=device
                    )
                    .detach()
                    .cpu()
                )

                if save_embeddings:
                    log.debug(f"Saving {embed_file}")
                    torch.save(rep, embed_file)

            except Exception as E:
                log.error(
                    f"Error: Unable to embed {_pdb}. Out of GPU memory? Consider using --cpu_only flag or setting --max_gpu_pdb_length lower (default 1000 residues)"
                )
                log.error(f"Error: {E}")

                if verbose >= 2:
                    traceback.print_exc()

                list_IF1_tensors.append(False)
                list_structures.append(False)
                list_sequences.append(False)
                continue

            # If everything succeeded, append values
            list_IF1_tensors.append(rep)
            list_structures.append(structure_full)
            list_sequences.append(seq)

    return list_IF1_tensors, list_structures, list_sequences


def get_basename_no_ext(filepath):
    """Returns file basename excluding extension, e.g. dir/5kja_A.pdb -> 5kja_A"""
    return os.path.splitext(os.path.basename(filepath))[0]


def load_structure_discotope(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        [biotite.structure.AtomArray]
    """

    if fpath[-3:] == "cif":
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure_full = pdbx.get_structure(pdbxf, model=1, extra_fields=["b_factor"])

    elif fpath[-3:] == "pdb":
        with open(fpath) as fin:
            try:
                pdbf = pdb.PDBFile.read(fin)
                structure_full = pdbf.get_structure(model=1, extra_fields=["b_factor"])
            except Exception as E:
                log.error(f"Unable to read PDB file {fpath}: {E}")

    # For IF1 embedding, only backbone is extracted (C, Ca, N atoms)
    bbmask = filter_backbone(structure_full)
    structure = structure_full[bbmask]

    # For "full" structure, extract all (amino-acid residue) atoms
    structure_full = structure_full[filter_amino_acids(structure_full)]

    # By default all chains are loaded, but only single chains are inputted to DiscoTope-3.0
    all_chains = get_chains(structure)

    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")

    if chain is None:
        chain_ids = all_chains

    elif isinstance(chain, list):
        chain_ids = chain

    else:
        chain_ids = [chain]

    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")

    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]

    chain_filter_full = [a.chain_id in chain_ids for a in structure_full]
    structure_full = structure_full[chain_filter_full]

    return structure, structure_full


def normalize_minmax(data):
    """Normalize to between 0 and 1"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def convert_resid_to_relative(res_id: np.ndarray) -> np.array:
    """
    Converts array of absolute indices to relative indices

    Input: biotite.structure.AtomArray.res_id:
        np.ndarray([5,5,5,6,6,7 ])
    Returns: np.array
        np.array([0,0,0,1,1,2])
    """

    # Get sorted list of unique, absolute indices
    res_idxs = sorted(np.unique(res_id))

    # Map absolute indices to relative
    d = {}
    for c, i in zip(res_idxs, range(len(res_idxs))):
        d[c] = i

    # Return as  np.array
    rel_idxs = np.array([d[v] for v in res_id])

    return rel_idxs


def embed_pdbs_IF1(
    pdb_dir: str,
    out_dir: str,
    struc_type: str,
    overwrite_embeddings=False,
    max_gpu_pdb_length: int = 1000,
    verbose: int = 1,
) -> None:
    """
    Embeds a directory of PDBs using IF1, either solved or AF2 structures (uses confidence)

    Input:
        pdb_dir: Folder with PDB files globbed with *.pdb
        out_dir: Directory to save tensor files
        struc_type: Solved for experimental PDBs (RCSB), "alphafold" for AlphaFold PDBs (alphafolddb)
        overwrite_embeddings: Overwrites previously computed tensor files if True
    Output:
        (None): Saves torch tensor .pt files with per-residue embeddings
    """

    import esm
    import esm.inverse_folding

    # Load ESM inverse folding model
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    pdb_paths = glob.glob(f"{pdb_dir}/*.pdb")

    exist_count = 0
    for i, pdb_path in enumerate(pdb_paths):
        pdb = os.path.splitext(os.path.basename(pdb_path))[0]
        outpath = f"{out_dir}/{pdb}.pt"

        if os.path.exists(outpath) and not overwrite_embeddings:
            if verbose:
                exist_count += 1
                # log.debug(f"Skipping embedding {pdb_path}, already exists")

        else:
            log.debug(f"{i+1} / {len(pdb_paths)}: Embedding {pdb_path} using ESM-IF1")

            chain_id = None

            # Extract representation
            try:
                structure, _ = load_structure_discotope(pdb_path, chain_id)
            except Exception as E:
                log.error(f"Unable to load structure, retrying with IF1 logic\n:{E}")
                structure = esm.inverse_folding.util.load_structure(
                    str(pdb_path), chain_id
                )

            coords, seq = esm_util_custom.extract_coords_from_structure(structure)

            if verbose:
                log.debug(f"PDB {pdb}, IF1: {len(seq)} residues")

            # Embed on CPU if PDB is too large (>= 1000 residues)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if len(seq) >= max_gpu_pdb_length:
                device = "cpu"

            # Save representation to file
            log.debug(f"ESM-IF1: Not including confidence")
            confidence = None
            rep = esm_util_custom.get_encoder_output(
                model, alphabet, coords, confidence, seq, device
            )
            torch.save(rep, outpath)

    log.debug(
        f"Skipped embedding {exist_count} / {len(pdb_paths)} PDBs (already exists)"
    )


def map_3letter_to_1letter(res_names: np.array):
    """Maps 3-letter to 1-letter residue names, with "X" for unknown residues."""

    mapping = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
    }

    # Map 3-letter to 1-letter residue names
    seq = pd.Series(res_names).map(mapping, na_action="ignore").values

    # Map unkonwn residues to 'X'
    seq[pd.isna(seq)] = "X"

    # Check for mismatches
    if "X" in np.unique(seq):
        # Get PDB id from outside function
        global pdb_id

        idxs = np.where(seq == "X")[0]
        unknown_res = np.unique(res_names[idxs])
        log.error(f"Unknown residues found in PDB {pdb_id}: {unknown_res}")

    return seq


def get_atomarray_seq_residx(
    atom_array: biotite.structure.AtomArray,
) -> "tuple[str, np.ndarray]":
    """
    Returns sequence and residue indices for an atom array
    """

    # Get residue indices and 3-letter encoding
    res_idxs, res_names = get_residues(atom_array)

    # Maps 3-letter to 1-letter residue names, unknown to X
    seq = map_3letter_to_1letter(res_names)

    return seq, res_idxs


def get_atomarray_res_sasa(atom_array):
    """Return per-residue SASA"""

    # Extract SASA
    # The following line calculates the atom-wise SASA of the atom array
    atom_sasa = biotite.structure.sasa(atom_array, vdw_radii="ProtOr")

    # Sum up SASA for each residue in atom array. Exclude nans with np.nansun
    res_sasa = biotite.structure.apply_residue_wise(atom_array, atom_sasa, np.nansum)

    return res_sasa


def get_atomarray_bfacs(atom_array: biotite.structure.AtomArray) -> np.array:
    """
    Return per-residue B-factors (AF2 confidence) from biotite.structure.AtomArray
    """

    # Get relative indices starting from 1. Copy to avoid modifying original
    atom_array_renum = biotite.structure.renumber_res_ids(
        copy.deepcopy(atom_array), start=1
    )

    # Extract B-factors, map to residues
    res_idxs_dict = {r: i for i, r in enumerate(atom_array_renum.res_id)}
    res_idxs_uniq = np.array(list(res_idxs_dict.values()))
    res_bfacs = atom_array_renum.b_factor[res_idxs_uniq]

    return res_bfacs


def structure_extract_seq_residx_bfac_rsas_diam(struc_full, chain_id):
    """
    Input:
        pdb_path: PDB file path (str)
    Returns:
        sequence: PDB extract residue sequence (str),
        res_idxs: PDB Residue absolute indices (np.array),
        bfacs: PDB B-factor filed, per residue,
        rsas: PDB SASA, calculated from Biotite
    """

    # pdbf = fastpdb.PDBFile.read(pdb_path)
    # structure = pdbf.get_structure(model=1, extra_fields=["b_factor"])
    # _, struc_full = load_structure_discotope(pdb_path, chain_id)

    seq, res_idxs = get_atomarray_seq_residx(struc_full)
    bfacs = get_atomarray_bfacs(struc_full)

    # Convert SASA to RSA with Sander scale max values
    sasa = get_atomarray_res_sasa(struc_full)
    div_values = [sasa_max_dict[aa] for aa in seq]
    rsa = sasa / div_values

    if len(rsa) != len(seq):
        log.error(
            f"Error: RSA values {len(rsa)} do not match length of backbone residues {len(seq)}"
        )

    return seq, res_idxs, bfacs, rsa


class Discotope_Dataset_web(torch.utils.data.Dataset):
    """
    Creates pre-processed torch Dataset directories with PDBs and IF1 tensors
    Note: PDB filenames and ESM-IF1 tensor names must be shared!

    Samples (antigens) are returned as dict, with keys X_arr, y_arr, length, pLDDT, feature_idxs etc ...
    X_arr is a torch tensor with features corresponding to (available from features_idxs key):
    - 0:512 = IF_tensors (512)
    - 512:532 = One-hot encoded amino-acids (20)
    - 532:533 = Residue Alphafold2 model pLDDT score (1)
    - 533:534 = Length of antigen, same for all residues (1)
    - 534:535 = Residue RSA, extracted using DSSP on Alphafold2 model (1)

    Args:
        pdb_dir: Directory with PDBs
        preprocess: Set to True to pre-process samples (True)
        n_jobs: Parallelize pre-processing across n cores (1)

    Example usage:
    dataset = Discotope_Dataset(pdb_dir)

    # Extract one sample (antigen)
    sample = dataset[0]
    X_arr = sample["X_arr"]
    y_arr = sample["y_arr"]

    # Get IF1 features only
    X_arr[:, sample["feature_idxs"]["IF1_tensor"]]

    Read more:
    - ESM-IF1 https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding
    - AlphaFold2 FAQ, pLDDT scores: https://alphafold.ebi.ac.uk/faq
    """

    def __init__(
        self,
        pdb_dir: str,
        structure_type: int,  # alphafold or solved
        check_existing_embeddings: bool = False,  # Try to load previous embedding files
        save_embeddings: bool = False,  # Save new embedding files
        preprocess: bool = True,
        cpu_only: bool = False,
        max_gpu_pdb_length: int = 1000,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> torch.utils.data.Dataset:
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.preprocess = preprocess
        self.structure_type = structure_type
        self.check_existing_embeddings = check_existing_embeddings
        self.save_embeddings = save_embeddings
        self.cpu_only = cpu_only
        self.max_gpu_pdb_length = max_gpu_pdb_length

        # Get PDB path dict
        self.list_pdb_files = glob.glob(f"{pdb_dir}/*.pdb")
        if len(self.list_pdb_files) == 0:
            log.error(f"No .pdb files found in {pdb_dir}")
            sys.exit(0)

        log.debug(f"Read {len(self.list_pdb_files)} PDBs from {pdb_dir}")

        if self.preprocess:
            # Read in IF1 tensors
            (
                self.list_IF1_tensors,
                self.list_structures,
                self.list_sequences,
            ) = load_IF1_tensors(
                self.list_pdb_files,
                check_existing_embeddings=self.check_existing_embeddings,
                save_embeddings=self.save_embeddings,
                cpu_only=self.cpu_only,
                max_gpu_pdb_length=self.max_gpu_pdb_length,
            )

            # Remaining pre-processing
            self.preprocessed_dataset = self.process_samples_parallel(
                n_jobs=self.n_jobs
            )

            # Filter out failed samples
            self.preprocessed_dataset = [d for d in self.preprocessed_dataset if d]

        else:
            self.preprocessed_dataset = []
            log.error(f"Must pre-process Discotope_Dataset")
            sys.exit()

    def one_hot_encode_sequence(self, seq: str):
        """One-hot encodes amino-acid sequence to 20 dims. Uses full 0 for unkown amino-acids"""
        seq = seq.upper()

        # Mapping to 21 dimensions, but the last will be excluded
        mapping = dict(zip("ACDEFGHIKLMNPQRSTVWYX", range(21)))

        # Map X and unknown to index 20 (last index in range 21)
        seq2 = [mapping.get(aa, 21 - 1) for aa in seq]

        # 21 dim one-hot -> 20 dim by excluding "X"
        oh = np.eye(21)[seq2]
        oh = oh[:, 0:20]

        return oh

    def one_hot_decode(self, one_hot):
        """De-codes one-hot encded sequences from 20 dims"""
        mapping = dict(zip(range(20), list("ACDEFGHIKLMNPQRSTVWY")))
        idxs = one_hot.argmax(axis=1)
        seq = [mapping[int(i)] for i in idxs]
        return seq

    def load_IF_tensor(self, pdb_id):
        """Load ESM inverse folding tensor"""
        if1_fp = f"{self.IF1_path}/{self.id_basename_dict[pdb_id]}.pt"

        return torch.load(if1_fp)

    def process_samples_parallel(self, n_jobs=8):
        """Processes samples in parallell"""

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.process_sample)(i) for i in range(len(self.list_pdb_files))
        )
        return results

    def process_sample(self, idx):
        """Process individual pdb, sequence, ESM"""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get PDB filepath from PDB id and PDB directory
        pdb_fp = self.list_pdb_files[idx]
        pdb_id = get_basename_no_ext(pdb_fp)
        log.debug(f"Processing {idx+1} / {len(self.list_pdb_files)}: PDB {pdb_id}")

        try:
            # Read variables
            IF1_tensor = self.list_IF1_tensors[idx]
            struc_full = self.list_structures[idx]
            seq = self.list_sequences[idx]

            if (
                type(IF1_tensor) is bool
                or type(struc_full) is bool
                or type(seq) is bool
            ):
                log.debug(f"Previously failed to pre-process {pdb_id}, skipping ...")
                return False

            seq_onehot = self.one_hot_encode_sequence(seq)
            L = len(seq)
            lengths = np.array([L] * L)

            # Extract values from PDB
            (
                pdb_seq,
                pdb_res_idxs,
                pdb_bfacs,
                pdb_rsas,
            ) = structure_extract_seq_residx_bfac_rsas_diam(struc_full, chain_id=None)

            # Struc_type 1 and pLDDTs set to B-factors only if AlphaFold structure
            if self.structure_type == "alphafold":
                struc_type = 1
            else:
                struc_type = 0
                pdb_bfacs = np.full(len(IF1_tensor), 100)

            # Feature tensor for training/prediction
            X_arr = np.concatenate(
                [
                    IF1_tensor.detach().numpy(),  # L x 512
                    seq_onehot,  # L x 20
                    pdb_bfacs.reshape(-1, 1),  # L x 1
                    lengths.reshape(-1, 1),  # L x 1
                    np.array([struc_type] * L).reshape(-1, 1),  # L x 1
                    pdb_rsas.reshape(-1, 1),  # L x 1
                ],
                axis=1,
            )

            # Dictionary with feature mappings
            feature_idxs = {
                "IF_tensor": range(0, 512),
                "sequence": range(512, 532),
                "pLDDTs": range(532, 533),
                "lengths": range(533, 534),
                "alphafold_struc_flag": range(534, 535),
                "RSAs": range(535, 536),
            }

            # DataFrame for easy overview
            df_stats = pd.DataFrame(
                {
                    "pdb": pdb_id,
                    "res_id": pdb_res_idxs,
                    "residue": list(seq),
                    "rsa": pdb_rsas.flatten(),
                    "pLDDTs": pdb_bfacs.flatten(),
                    "length": lengths.flatten(),
                    "alphafold_struc_flag": struc_type,
                }
            )

            output_dict = {
                "pdb_id": pdb_id,
                "pdb_fp": pdb_fp,
                "X_arr": X_arr,
                "df_stats": df_stats,
                "length": L,
                "pLDDTs": pdb_bfacs,
                "RSAs": pdb_rsas,
                "sequence_str": seq,
                "pdb_seq": pdb_seq,
                "feature_idxs": feature_idxs,
                "alphafold_struc_flag": struc_type,
                "PDB_biotite": struc_full,
            }

            return output_dict

        except Exception as E:
            log.error(f"Error processing chain {pdb_id}: {E}")

            if self.verbose >= 2:
                traceback.print_exc()

            return False

    def __len__(self):
        return len(self.preprocessed_dataset)

    def __getitem__(self, idx):
        """Returns sample dict from pre-processed dataset"""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.preprocessed_dataset[idx]


def main(args):
    """Main function"""

    # Make sure output directories exist
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_embeddings:
        log.debug(f"Loading ESM-IF1 to embed PDBs")
        embed_pdbs_IF1(
            args.pdb_dir,
            out_dir=args.pdb_dir,
            struc_type=args.struc_type,
            overwrite_embeddings=args.overwrite_embeddings,
            max_gpu_pdb_length=args.max_gpu_pdb_length,
            verbose=args.verbose,
        )

    log.debug(f"Pre-processing PDBs")
    dataset = Discotope_Dataset_web(
        pdb_dir=args.pdb_dir, structure_type=args.struc_type, verbose=args.verbose
    )
    log.info(
        f"Writing processed dataset ({len(dataset)} samples) to {f'{args.out_dir}/dataset.pt'}"
    )
    torch.save(dataset, f"{args.out_dir}/dataset.pt")

    log.debug(f"Done!")


if __name__ == "__main__":
    args = cmdline_args()

    # Log if running as main
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[{asctime}] {message}",
        style="{",
        handlers=[
            logging.FileHandler(f"{args.out_dir}/log.log"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    log.info("Making Discotope-3.0 dataset")
    log.info("Using Sander residue max ASA values")

    main(args)
