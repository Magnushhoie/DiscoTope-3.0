import glob
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
    output = subprocess.run(bashCommand, timeout=timeout, capture_output=True, shell=True)
    if print_out:
        print(output.stdout.decode("utf-8"))


while True:
    os.system("cls" if os.name == "nt" else "clear")
    msg(
        rf"""
        Choose an option:
        0) autolint
        1) predict full (solved)
        2) predict chains (solved)
        3) predict chains (alphafold)
        4) predict list file
        5) predict single PDB (fetch)
        """
    )

    choice = input("Choice: ")

    if selected(choice, "0", "autolint"):
        print(f"Running black, isort and flake8 on {dir}")
        run_bash(f"black {dir} && isort {dir} && flake8 {dir}")

    if selected(choice, "1", "predict full solved"):
        msg(
            rf"""
        python src/predict_webserver.py \
        --list_file data/test_full_solved_list.txt \
        --list_id rcsb --struc_type solved \
        --out_dir job_out/test_full \
        {verbose}
        """
        )

    if selected(choice, "2", "predict chains (solved)"):
        msg(
            rf"""
        python src/predict_webserver.py \
        --pdb_dir data/test_af2 \
        --struc_type solved \
        --out_dir job_out/test_full \
        {verbose}
        """
        )

    if selected(choice, "3", "predict chains (alphafold)"):
        msg(
            rf"""
        python src/predict_webserver.py \
        """
        )

    if selected(choice, "4", "predict list file"):
        list_files = glob.glob(f"data/*.txt")
        for i, list_file in enumerate(list_files):
            print(f"{i}) {list_file}")
        idx = int(input("Pick file no: "))
        list_file = list_files[idx]

        msg(
            rf"""
        python src/predict_webserver.py \
        --list_file {list_file} \
        --list_id rcsb --struc_type solved \
        --out_dir job_out/test_full \
        {verbose} 
        """
        )

    if selected(choice, "5", "predict single PDB (fetch)"):
        pdb_id = input("Enter PDB ID: ")
        with (open(f"data/{pdb_id}.txt", "w")) as f:
            f.write(pdb_id)

        struc_type = input("Enter PDB type (alphafold or rcsb): ")

        msg(
            rf"""
        python src/predict_webserver.py \
        --list_file data/{pdb_id}.txt \
        --list_id rcsb --struc_type {struc_type} \
        --out_dir job_out/{pdb_id} \
        {verbose}
        """
        )

    # End
    input("Run command and press enter when done.")
