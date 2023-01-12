import glob
import os
import subprocess
import textwrap

dir = "."
verbose = "--verbose 2"


def msg(message):
    """Print and dedent"""
    print(textwrap.dedent(message))


def selected(choice, opt1, opt2):
    return choice in [opt1, opt2]


def run_bash(bashCommand, print_out=True, timeout=10):
    output = subprocess.run(bashCommand, timeout=10, capture_output=True, shell=True)
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
        4) Predict list file
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

    # End
    input("Run command and press enter when done.")
