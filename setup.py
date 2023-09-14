from setuptools import setup, find_packages
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name='discotope3',
    version='1.0',    
    packages=find_packages(),
    description='B-cell epitope prediction using inverse folding embeddings',
    url='https://github.com/Magnushhoie/discotope3_web/',
    author='Magnus Haraldson Høie',
    author_email='maghoi@dtu.dk',
    license='N/A',
    install_requires=REQUIREMENTS,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9',
    ],
)
