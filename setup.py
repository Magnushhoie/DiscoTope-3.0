from setuptools import find_packages, setup

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="discotope3",
    version="1.0",
    packages=find_packages(),
    description="Protein structure B-cell epitope prediction using inverse folding representations",
    url="https://github.com/Magnushhoie/discotope3_web/",
    author="Magnus Haraldson Høie",
    author_email="maghoi@dtu.dk",
    license="CC BY-NC 4.0",
    install_requires=REQUIREMENTS,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.9",
    ],
)
