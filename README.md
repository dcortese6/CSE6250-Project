# CSE6250-Project

## Initialize Environment
```conda env create -f environment.yml```

## Download MIMIC-III Data

https://physionet.org/content/mimiciii/1.4/

Insert the following files into the directory `data/codes/`:

    - CPTEVENTS.csv
    - DIAGNOSES_ICD.csv
    - PROCEDURES_ICD.csv

Insert the following files into the directory `data/notes/`:

    - NOTEEVENTS.csv

**NOTE:** The above files will be downloaded as `.csv.gz` files. You can use `gunzip` in your terminal to unzip them if you have a Mac.

Be sure to have empty directories `data/lookups/` and `data/model/` as structured in the repo.

## Prepare Dataset

```python preprocess.py```

This will take a few minutes to run. The prepared dataset will be saved as pickle files in `data/model/`.

## Train Model

**IN PROGRESS**