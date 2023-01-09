import glob
import logging

logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
log = logging.getLogger(__name__)

import os
import re
import sys
# Set project path two levels up
from pathlib import Path

import Bio
import biotite.structure
import fastpdb
import numpy as np
import pandas as pd
import prody
import torch
from Bio import SeqIO
from biotite.sequence import ProteinSequence
from biotite.structure import filter_backbone, get_chains
from biotite.structure.io import pdb, pdbx
from biotite.structure.residues import get_residues
from joblib import Parallel, delayed

# ROOT_Path 2 levels up
ROOT_PATH = str(Path(os.getcwd()))
sys.path.insert(0, ROOT_PATH)

from argparse import ArgumentParser, RawTextHelpFormatter

# Use Sander values instead
from Bio.Data.PDBData import residue_sasa_scales
from Bio.SeqUtils import seq1

import src.esm_util_custom

# Sander or Wilke
# residue_max_acc_1seq = {seq1(aa) : residue_sasa_scales["Sander"][aa] for aa, val in residue_sasa_scales["Sander"].items()}
# sasa_max_dict = residue_max_acc_1seq

residue_max_acc_1seq = {
    seq1(aa): residue_sasa_scales["Sander"][aa]
    for aa, val in residue_sasa_scales["Sander"].items()
}
sasa_max_dict = residue_max_acc_1seq


def cmdline_args():
    # Make parser object
    usage = f"""
    # Make dataset with test set
    python src/data/make_dataset.py \
    --fasta data/raw/external_testset_bp3.fasta \
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
        "--fasta",
        required=True,
        help="Input FASTA file, may contain epitopes in uppercase",
        metavar="FILE",
        type=lambda x: is_valid_path(p, x),
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
                # Try fast processing
                pdbf = fastpdb.PDBFile.read(fin)
                structure_full = pdbf.get_structure(model=1, extra_fields=["b_factor"])
            except Exception as E:
                log.info(
                    f"Unable to load PDB with fastpdb, retrying with Biotite:\n{E}"
                )
                pdbf = pdb.PDBFile.read(fin)
                structure_full = pdbf.get_structure(model=1, extra_fields=["b_factor"])

    bbmask = filter_backbone(structure_full)
    structure = structure_full[bbmask]

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


def save_fasta_from_pdbs(pdb_dir: str, out_dir: str):
    """Reads in sequences from PDBs from pdb_dir, saves as pdbs.fasta in out_dir"""

    def prody_read_seq(pdb_path: str):
        """Reads sequence from PDB, based on carbon-alpha atoms"""

        structure = prody.parsePDB(pdb_path, subset="ca")

        for chain in np.unique(structure.getChids()):
            seq = structure[chain].getSequence()
            yield chain, seq

    pdbs = list(Path(pdb_dir).glob("*.pdb"))
    fasta_dict = {}

    for pdb_path in pdbs:
        pdb = os.path.splitext(os.path.basename(pdb_path))[0]

        for chain, seq in prody_read_seq(str(pdb_path)):
            id = f"{pdb}_{chain}"

            fasta_dict[id] = SeqIO.SeqRecord(
                Bio.Seq.Seq(seq), id=id, name=id, description=id
            )

    # write FASTA
    outfile = f"{out_dir}/pdbs.fasta"
    log.info(f"Writing {len(fasta_dict)} FASTA entries to {outfile}")
    with open(outfile, "w") as out_handle:
        SeqIO.write(fasta_dict.values(), out_handle, "fasta")


def embed_pdbs_IF1(
    pdb_dir: str,
    out_dir: str,
    input_fasta: dict[str, "Bio.SeqRecord.SeqRecord"],
    struc_type: str,
    overwrite_embeddings=False,
    verbose: int = 1,
) -> None:
    """
    Embeds a directory of PDBs using IF1, either solved or AF2 structures (uses confidence)

    Input:
        pdb_dir: Folder with PDB files matching PDB ids in input_fasta,
                globbed with *.pdb. Must specify whether they are solved or predicted
        out_dir: Directory to save tensor files
        input_fasta: FASTA dictionary containing PDB ids as keys (e.g. 4akj_C), used to select which
                PDBs to embed
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

        # if pdb not in input_fasta:
        #    print(f"Skipping {pdb}, not in input_fasta")
        #    continue

        if os.path.exists(outpath) and not overwrite_embeddings:
            if verbose:
                exist_count += 1
                # log.info(f"Skipping embedding {pdb_path}, already exists")

        else:
            log.info(f"{i+1} / {len(pdb_paths)}: Embedding {pdb_path} using ESM-IF1")

            # MH: Better system for choosing chains needed
            # Extract chain from last character after "_"
            # chain_id = "A"
            # chain_id = pdb.split("_")[1][0]

            # if struc_type != "solved":
            # chain_id = None
            chain_id = None

            # Extract representation
            # structure = esm.inverse_folding.util.load_structure(str(pdb_path), chain_id)
            try:
                structure, _ = load_structure_discotope(pdb_path, chain_id)
            except Exception as E:
                log.info(f"Unable to load structure, retrying with IF1 logic\n:{E}")
                structure = esm.inverse_folding.util.load_structure(
                    str(pdb_path), chain_id
                )

            coords, seq = src.esm_util_custom.extract_coords_from_structure(structure)

            # Extra
            # seq_idxs = np.unique(structure.res_id)
            # if struc_type == "alphafold" and not exclude_conf:
            #    log.info(f"Including pLDDTs in IF1 embedding")
            #    confidence = get_atomarray_bfacs(structure)
            # else:

            if verbose:
                log.info(f"PDB {pdb}, IF1: {len(seq)} residues")

            # Check whether to put on GUP
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if len(seq) >= 1000:
                device = "cpu"

            # Save representation to file
            log.info(f"ESM-IF1: Not including confidence")
            confidence = None
            rep = src.esm_util_custom.get_encoder_output(
                model, alphabet, coords, confidence, seq, device
            )
            torch.save(rep, outpath)

    log.info(
        f"Skipped embedding {exist_count} / {len(pdb_paths)} PDBs (already exists)"
    )


def get_atomarray_seq_residx(atom_array):
    """Return sequence and residue indices"""

    res_idxs, res_names = get_residues(atom_array)
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in res_names])

    return seq, res_idxs


def get_atomarray_res_sasa(atom_array):
    """Return per-residue SASA"""

    # Extract SASA
    # The following line calculates the atom-wise SASA of the atom array
    atom_sasa = biotite.structure.sasa(atom_array, vdw_radii="ProtOr")
    # Sum up SASA for each residue in atom array
    res_sasa = biotite.structure.apply_residue_wise(atom_array, atom_sasa, np.sum)

    return res_sasa


def get_atomarray_bfacs(atom_array: biotite.structure.AtomArray) -> np.array:
    """
    Return per-residue B-factors (AF2 confidence) from biotite.structure.AtomArray
    """

    # Extract B-factors
    res_idxs_dict = {r: i for i, r in enumerate(atom_array.res_id)}
    res_idxs_uniq = np.array(list(res_idxs_dict.values()))

    # Assign to series with assigned residue indices
    res_bfacs = atom_array.b_factor[res_idxs_uniq]

    return res_bfacs


def get_atomarray_diameter_gyrrad(atom_array):

    """
    Return diameter
    # https://www.biotite-python.org/examples/gallery/structure/diameter.html#sphx-glr-examples-gallery-structure-diameter-py
    """

    # Radius of gyration
    try:
        # Remove all non-amino acids
        atom_array = atom_array[biotite.structure.filter_amino_acids(atom_array)]
        coord = atom_array.coord
        # Calculate all pairwise difference vectors
        diff = coord[:, np.newaxis, :] - coord[np.newaxis, :, :]
        # Calculate absolute of difference vectors -> square distances
        sq_dist = np.sum(diff * diff, axis=-1)
        # Maximum distance is diameter
        diameter = np.sqrt(np.max(sq_dist))

        gyr_rad = biotite.structure.gyration_radius(atom_array)
    except Exception as E:
        gyr_rad = np.nan
        log.error(f"Unable to calculate gyration radius:\n{E}")

    return diameter, gyr_rad


def pdb_extract_seq_residx_bfac_rsas_diam(pdb_path, chain_id):
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
    _, struc_full = load_structure_discotope(pdb_path, chain_id)

    seq, res_idxs = get_atomarray_seq_residx(struc_full)

    bfacs = get_atomarray_bfacs(struc_full)
    diam, gyr_rad = get_atomarray_diameter_gyrrad(struc_full)

    # Convert sasa to rsa
    sasa = get_atomarray_res_sasa(struc_full)
    seq = seq.upper()
    div_values = [sasa_max_dict[aa] for aa in seq]
    rsa = sasa / div_values

    return seq, res_idxs, bfacs, rsa, diam, gyr_rad


class Discotope_Dataset(torch.utils.data.Dataset):
    """
    Creates pre-processed torch Dataset from input FASTA dict and directories with PDBs and IF-1 tensors
    Note: FASTA IDs, PDB filenames and ESM-IF1 tensor names must be shared!

    Samples (antigens) are returned as dict, with keys X_arr, y_arr, length, pLDDT, feature_idxs etc ...
    X_arr is a torch tensor with features corresponding to (available from features_idxs key):
    - 0:512 = IF_tensors (512)
    - 512:532 = One-hot encoded amino-acids (20)
    - 532:533 = Residue Alphafold2 model pLDDT score (1)
    - 533:534 = Length of antigen, same for all residues (1)
    - 534:535 = Residue RSA, extracted using DSSP on Alphafold2 model (1)

    Args:
        fasta_dict: Dictionary from SeqIO.to_dict(SeqIO.parse(input_fasta, "fasta"))
        pdb_dir: Directory with PDBs, shared IDs are FASTA IDs
        IF1_dir: Directory with ESM-IF1 tensors (see Read More)
        preprocess: Set to True to pre-process samples (True)
        n_jobs: Parallelize pre-processing across n cores (1)

    Example usage:
    fasta_dict = SeqIO.to_dict(SeqIO.parse(input_fasta, "fasta"))
    dataset = Discotope_Dataset(fasta_dict, pdb_dir, IF1_dir)

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
        fasta_dict: "dict[str, Bio.SeqIO.SeqRecord]",
        pdb_dir: str,
        IF1_dir: str,
        preprocess: bool = True,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> torch.utils.data.Dataset:

        self.verbose = verbose

        # Input FASTA
        self.fasta_dict = fasta_dict

        # Directory with AlphaFold2 PDBs
        self.PDB_path = Path(pdb_dir)

        # Directory with pre-processed ESM-IF1 embeddings
        self.IF1_path = Path(IF1_dir)

        # Convert FASTA IDs to sorted list
        self.key_list = sorted(fasta_dict.keys())

        # Parallell jobs to run while preprocessing (don't use)
        self.n_jobs = n_jobs

        # preprocess
        self.preprocess = preprocess

        # Dictionary mapping FASTA/PDB IDs to basenames for retrieving .pdb and .pt files
        self.id_basename_dict = self.create_id_with_basename_dict(self.PDB_path)

        self.key_list = list(self.id_basename_dict.keys())

        # Check that all FASTA IDs are found among PDB IDs
        missing_ids = set(self.key_list) - set(self.id_basename_dict.keys())
        if len(missing_ids) >= 1:
            log.error(
                f"Unable to find all FASTA IDs in pdb_dir {pdb_dir}:\n{missing_ids}"
            )
            sys.exit()

        if self.preprocess:
            results = self.process_samples_parallel(n_jobs=self.n_jobs)

            # Exclude results if not expected returned dict
            missing = [r for r in results if type(r) != dict]
            results = [r for r in results if type(r) == dict]

            if len(missing) >= 1:
                print(f"Unable to process {len(missing)} PDBs:\n{missing}\n")

            self.preprocessed_dataset = results
        else:
            self.preprocessed_dataset = []
            log.error(f"Must pre-process Discotope_Dataset with")
            sys.exit()

    def create_id_with_basename_dict(
        self,
        pdb_dir: str,
        verbose: int = 0,
    ) -> "dict[str : str]":
        r"""
        Create dict to match PDB ID with PDB filename, for later extracting id.pdb and id.pt files from FASTA IDs

        Our assumption of matching FASTA ID with PDB filename breaks for AlphaFold2 generated PDBs
        both IDs 6vtw_A and 6vtw_A_unrelaxed_rank_1_model_5 should match
        ../webserver/debug_pdbs/6vtw_A_unrelaxed_rank_1_model_5.pdb

        Returns:
            id_dict: dict matching PDB's ID with its basename, removing AlphaFold pattern r'_unrelaxed_rank_\d+_model_\d+'
        """

        # Find all PDB files in pdb_dir
        pdb_fullpaths = glob.glob(f"{pdb_dir}/*.pdb")

        id_dict = {}
        for pdb_path in pdb_fullpaths:

            # Extract PDB basename without file extension
            pdb_basename = os.path.splitext(os.path.basename(pdb_path))[0]

            # If AlphaFold2 pattern found, remove it from the ID
            id = re.sub(r"_unrelaxed_rank_\d+_model_\d+", "", pdb_basename)
            id = re.sub(r"_rank_\d+_model_\d+.*unrelaxed$", "", id)

            if id != pdb_basename and verbose:
                print(f"Removing AlphaFold2 pattern: {pdb_basename} -> {id}")

            # Match ID shortname and full basename
            id_dict[id] = pdb_basename
            id_dict[pdb_basename] = pdb_basename

        return id_dict

    def find_epitopes(self, seq: str):
        """Returns torch Tensor with 1 where residue is uppercase, 0 where residue is lowercase"""
        # return torch.Tensor([1 if res.isupper() else 0 for res in seq]).to(torch.int64)
        return np.array([1 if res.isupper() else 0 for res in seq])

    def one_hot_encode(self, seq: str):
        """One-hot encodes amino-acids to 20 dims"""
        seq = seq.upper()
        mapping = dict(zip("ACDEFGHIKLMNPQRSTVWY", range(20)))
        seq2 = [mapping[i] for i in seq]
        return np.eye(20)[seq2]

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
            delayed(self.process_sample)(i) for i in range(len(self.key_list))
        )
        return results

    def process_sample(self, idx):
        """Process individual pdb, fasta sequence, ESM"""

        log.info(f"Processing sample {idx+1} of {len(self.fasta_dict)}")

        if torch.is_tensor(idx):
            idx = idx.tolist()

        pdb_id = self.key_list[idx]

        # Get PDB filepath from PDB id and PDB directory
        pdb_fp = f"{self.PDB_path}/{self.id_basename_dict[pdb_id]}.pdb"
        seq_str = prody.parsePDB(pdb_fp, subset="ca").getSequence()
        # seq_str = str(self.fasta_dict[pdb_id].seq)

        if self.verbose:
            log.info(
                f"PDB id {pdb_id}, FASTA {len(seq_str)} residues, PDB path {pdb_fp}"
            )

        try:

            # Add sequence from FASTA
            seq_onehot = torch.from_numpy(self.one_hot_encode(seq_str))

            # Get antigen length
            L = len(seq_str)
            lengths = torch.tensor([L] * L)

            # Extract values from PDB
            (
                pdb_seq,
                pdb_res_idxs,
                pdb_bfacs,
                pdb_rsas,
                pdb_diam,
                pdb_gyrrad,
            ) = pdb_extract_seq_residx_bfac_rsas_diam(pdb_fp, chain_id=None)

            if pdb_seq.upper() != seq_str.upper():
                log.error(f"Mismatch between PDB sequence and FASTA sequence")
                print(pdb_seq.upper())
                print(seq_str.upper())
                raise Exception

            # Load inverse folding tensor
            IF_tensor = self.load_IF_tensor(pdb_id)

            # Dictionary with feature mappings
            feature_idxs = {
                "IF_tensor": range(0, 512),
                "sequence": range(512, 532),
                "pLDDTs": range(532, 533),
                "lengths": range(533, 534),
                "struc_type": range(534, 535),
                "RSAs": range(535, 536),
            }

            # Infer structure type by B-factor column
            if (pdb_bfacs == 100).sum() == len(pdb_bfacs):
                # Solved if B-factor is all 100, forcibly set all residues to 100 at actual length
                struc_type = 0
                pdb_bfacs = np.full(len(IF_tensor), 100)

                if self.verbose:
                    log.info(f"Solved structure inferred")
            else:
                # AF2 Predicted if B-factor is not all 1
                struc_type = 1

                if self.verbose:
                    log.info(f"AF2 structure inferred")

            # log.info("No features")

            # Feature tensor for training/prediction
            X_arr = np.concatenate(
                [
                    IF_tensor.detach().numpy(),  # L x 512
                    seq_onehot,  # L x 20
                    pdb_bfacs.reshape(-1, 1),  # L x 1
                    lengths.reshape(-1, 1),  # L x 1
                    np.array([struc_type] * L).reshape(-1, 1),  # L x 1
                    pdb_rsas.reshape(-1, 1),  # L x 1
                ],
                axis=1,
            )

            # Target tensor for training, find epitopes from upper-case letters in sequence
            # NB: Ignore if residues are extracted from PDB sequence (returns all uppercase)
            y_arr = self.find_epitopes(seq_str)

            # import pdb; pdb.set_trace()

            # DataFrame for easy overview
            df_stats = pd.DataFrame(
                {
                    "pdb": pdb_id,
                    "idx": list(range(1, L + 1)),
                    "residue": list(seq_str),
                    "res_idx": pdb_res_idxs,
                    "rsa": pdb_rsas.flatten(),
                    "diam": pdb_diam,
                    "gyrrad": pdb_gyrrad,
                    "pLDDTs": pdb_bfacs.flatten(),
                    "length": lengths.flatten(),
                    "struc_type": struc_type,
                }
            )

            output_dict = {
                "pdb_id": pdb_id,
                "pdb_fp": pdb_fp,
                "X_arr": X_arr,
                "y_arr": y_arr,
                "df_stats": df_stats,
                "length": L,
                "pLDDTs": pdb_bfacs,
                "RSAs": pdb_rsas,
                "sequence_str": seq_str,
                "pdb_seq": pdb_seq,
                "feature_idxs": feature_idxs,
                "struc_type": struc_type,
            }

            return output_dict

        except Exception as E:
            log.error(f"PDB id {pdb_id}, length {len(seq_str)} from {pdb_fp}:\n{E}")
            import traceback

            traceback.print_exc()
            return pdb_id
        #    #raise Exception

    def __len__(self):
        return len(self.preprocessed_dataset)

    def __getitem__(self, idx):
        """Returns sample dict from pre-processed dataset"""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.preprocessed_dataset[idx]


def main(args):
    """Main function"""

    # If models directory not changed from default, assign as basename of out_dir
    # if args.models_path == f"{ROOT_PATH}/models/":
    #    args.models_path = f"{ROOT_PATH}/models/{os.path.basename(args.out_dir)}"
    #    log.info(f"Will save trained models to {args.models_path}")

    # Make sure output directories exist
    os.makedirs(args.out_dir, exist_ok=True)
    # os.makedirs(args.models_path, exist_ok=True)

    if args.fasta:
        if os.path.exists(args.fasta):
            log.info(f"Using provided FASTA file: {args.fasta}")
            input_fasta = args.fasta
    else:
        log.info(f"Creating FASTA file from PDB sequences: {args.out_dir}/pdbs.fasta")
        save_fasta_from_pdbs(args.pdb_dir, args.out_dir)
        input_fasta = f"{args.out_dir}/pdbs.fasta"

    # Load provided/created FASTA file and write again to args.out_dir
    fasta_dict = SeqIO.to_dict(SeqIO.parse(input_fasta, "fasta"))
    log.info(f"Read {len(fasta_dict)} entries from {input_fasta}")

    with open(f"{args.out_dir}/pdbs.fasta", "w") as out_handle:
        SeqIO.write(fasta_dict.values(), out_handle, "fasta")

    if not args.skip_embeddings:
        log.info(f"Loading ESM-IF1 to embed PDBs")
        embed_pdbs_IF1(
            args.pdb_dir,
            out_dir=args.pdb_dir,
            struc_type=args.struc_type,
            input_fasta=fasta_dict,
            overwrite_embeddings=args.overwrite_embeddings,
            verbose=args.verbose,
        )

    log.info(f"Pre-processing PDBs")
    dataset = Discotope_Dataset(
        fasta_dict, args.pdb_dir, IF1_dir=args.pdb_dir, verbose=args.verbose
    )
    log.info(
        f"Writing processed dataset ({len(dataset)} samples) to {f'{args.out_dir}/dataset.pt'}"
    )
    torch.save(dataset, f"{args.out_dir}/dataset.pt")

    log.info(f"Done!")


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
