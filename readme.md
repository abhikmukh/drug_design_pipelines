# Molecule design pipeline using Prolif and Rdkit Brics

## Overview
This repository contains a method to generate Protein ligand interaction fingerprint using Prolif and a method to design novel molecules using Rdkit Brics.

## Installation
After unzipping the folder please run this command to install the required packages
```
conda env create -n new-env-name -f environment.yml
```
## File structure
pipeline.ipynb notebook has the full workflow. The data folder contains the input files. The output folder contains the output files.
utils.py contains the helper functions.

## Usage
Please run the pipeline.ipynb notebook to see the full workflow. The notebook contains the detailed explanation of the workflow.
To make it run faster, please change the number_of_molecules to generate in design_new_molecules function
