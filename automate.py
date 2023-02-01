import os
import subprocess
import textwrap

dir = "src/"
verbose = "--verbose 2"


def msg(message):
    """Print and dedent"""
    print(textwrap.dedent(message))


def selected(choice, opt1, opt2):
    return choice in [opt1, opt2]


def run_bash(bashCommand, print_out=True, timeout=20):
    output = subprocess.run(
        bashCommand, timeout=timeout, capture_output=True, shell=True
    )
    if print_out:
        print(output.stdout.decode("utf-8"))


while True:
    os.system("cls" if os.name == "nt" else "clear")
    msg(
        rf"""
        Choose an option:
        1) Predict single PDB ID (fetch)
        2) Predict list file of PDB IDs (fetch)
        3) Predict compressed zip file containing PDB files (local)
        4) Predict single PDB file (local)
        5) Predict folder containing PDBs files (local)
        """
    )

    choice = input("Choice: ")

    if selected(choice, "1", "predict single PDB (fetch)"):

        pdb_id = input(
            "Enter PDB ID to fetch (e.g. 7k7i (RCSB) or F6KCZ5 (AlphaFoldDB): "
        )
        with (open(f"data/{pdb_id}.txt", "w")) as f:
            f.write(pdb_id)

        # Initial alternatives
        struc_type = input("Enter PDB type (alphafold or solved): ")
        if struc_type not in ["alphafold", "solved"]:
            raise ValueError("Invalid struc_type, please choose alphafold or solved.")

        msg(
            rf"""
        python src/predict_webserver.py \
        --list_file data/{pdb_id}.txt \
        --struc_type {struc_type} \
        --out_dir job_out/{pdb_id} \
        {verbose}
        """
        )

    if selected(choice, "2", "predict list file of PDB IDs"):

        # Initial alternatives
        struc_type = input("Enter PDB type (alphafold or solved): ")
        if struc_type not in ["alphafold", "solved"]:
            raise ValueError("Invalid struc_type, please choose alphafold or solved.")

        list_file = (
            "af2_list_uniprot.txt" if struc_type == "alphafold" else "solved_list.txt"
        )

        msg(
            rf"""
        python src/predict_webserver.py \
        --list_file data/{list_file} \
        --struc_type {struc_type} \
        --out_dir job_out/{list_file} \
        {verbose}
        """
        )

    if selected(
        choice, "3", "Predict compressed zip file containing PDB files (local)"
    ):

        msg(
            rf"""
        python src/predict_webserver.py \
        --pdb_or_zip_file data/single_solved.zip \
        --struc_type solved \
        --out_dir job_out/single_solved \
        {verbose} 
        """
        )

    if selected(choice, "4", "Predict single PDB ID (local)"):

        msg(
            rf"""
        python src/predict_webserver.py \
        --pdb_or_zip_file data/1L36.pdb \
        --struc_type solved \
        --out_dir job_out/1L36 \
        {verbose} 
        """
        )

    if selected(choice, "5", "Predict single PDB ID (local)"):

        msg(
            rf"""
        python src/predict_webserver.py \
        --pdb_dir data/ \
        --struc_type solved \
        --out_dir job_out/data \
        {verbose} 
        """
        )

    # End
    input("Run command and press enter when done.")
