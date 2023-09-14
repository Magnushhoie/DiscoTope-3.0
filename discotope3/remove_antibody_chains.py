import argparse
import glob
import os
import re
from pathlib import Path

from Bio.PDB import PDBIO, Select, parse_pdb_header
from Bio.PDB.PDBParser import PDBParser


def find_antibody_chains(pdb_path, verbose=False):
    """Finds antibody chains from a PDB structure."""

    keywords = [
        "antibody",
        "immunoglobulin",
        "fab",
        "heavy chain",
        "light chain",
        "heavychain",
        "lightchain",
        "variable domain",
    ]

    # Parse PDB header for "compound" column containing chain names
    header = parse_pdb_header(pdb_path)
    _pdb_name = Path(pdb_path).stem

    # Add to antibody chain list if keywords found
    try:
        ab_chains = []
        for v in header["compound"].values():
            if any([kw in v["molecule"].lower() for kw in keywords]):
                # Regex extract all possible single-letter chain names
                chains = v["chain"]
                matches = re.findall(r"[a-zA-Z]{1}", chains)

                assert len(matches) >= 1
                ab_chains.extend(matches)

        if verbose:
            print(f"Found antibody chains for PBD {_pdb_name}: {ab_chains}")

    # If unable to read, raise exception
    except Exception as E:
        print(f"Unable to extract pattern for {_pdb_name}: {E}")
        raise Exception

    return ab_chains


def save_clean_pdb_single_chains(pdb_path, pdb_name, bscore, outdir, verbose=True):
    """
    Function to save cleaned PDB file(s) with specified B-factor score.

    Parameters
    ----------
    pdb_path : str
        The path to the PDB file.
    pdb_name : str
        The name of the PDB file.
    bscore : 100 or None
        The B-factor score to be assigned to the residues/atoms, no change with None. 100 for solved structure, none for AF2
    outdir : str
        The directory to save the output file(s).
    save_full_complex : bool, optional
        If True, the function will save the whole complex as single PDB file.
        Else, individual chains are saved as separate PDB files. Default is False.
    """

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
            # Exclude antibody chains
            if chain.get_id().lower() in ab_chains:
                print(f"Removing ab chain {chain} for {pdb_name}")
                return False
            else:
                return self.chain is None or chain == self.chain

        def accept_atom(self, atom):
            if self.const_score is None:
                pass
            elif self.const_score:
                print(f"Setting B-factor to {self.bscore} for {pdb_name}")
                atom.set_bfactor(self.bscore)
            else:
                self.letter = atom.get_full_id()[3][2]
                if atom.get_full_id()[3][2] not in (self.prev_letter, " "):
                    print(
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

    # Start
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

    # Read antibody chains
    ab_chains = find_antibody_chains(pdb_path, verbose=verbose)

    # Save whole complex, cleaned
    pdb_out = f"{outdir}/{pdb_name}_noab.pdb"
    io_w_no_h = PDBIO()
    io_w_no_h.set_structure(structure)
    with open(pdb_out, "w") as f:
        print(*header, sep="\n", file=f)
        io_w_no_h.save(f, Clean_Chain(bscore, chain=None))


def main(input_folder, output_folder, verbose=True):
    pdb_files = glob.glob(f"{input_folder}/*.pdb")
    print(f"Found {len(pdb_files)} PDB files in {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    for _pdb in pdb_files:
        pdb_name = Path(_pdb).stem
        save_clean_pdb_single_chains(
            _pdb, pdb_name, bscore=None, outdir=output_folder, verbose=verbose
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove antibody chains from PDB files"
    )
    parser.add_argument(
        "--pdbdir", help="Path to the folder containing input PDB files"
    )
    parser.add_argument(
        "--outdir", help="Path to the folder where the output PDB files will be saved"
    )
    parser.add_argument(
        "-v", "--verbose", default=0, help="Display additional information"
    )

    args = parser.parse_args()

    main(args.pdbdir, args.outdir, args.verbose)


"""
pdb_dir = "../output/test_solved_full_complex/input_chains/"
pdb_files = glob.glob(f"{pdb_dir}/*.pdb")

for _pdb in pdb_files: 
    ab_chains = find_antibody_chains(_pdb)

for _pdb in pdb_files:
    pdb_name = Path(_pdb).stem
    save_clean_pdb_single_chains(_pdb, pdb_name, bscore=None, outdir="temp/")
"""
