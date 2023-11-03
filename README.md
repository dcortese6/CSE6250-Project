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

Be sure to have empty directories `data/lookups/` and `data/model/`. Your folder structure should look like this before preparing the dataset:

* [data/](./CSE6250-Project/data)
  * [codes/](./CSE6250-Project/data/codes)
    * [CPTEVENTS.csv](./CSE6250-Project/data/codes/CPTEVENTS.csv)
    * [DIAGNOSES_ICD.csv](./CSE6250-Project/data/codes/DIAGNOSES_ICD.csv)
    * [PROCEDURES_ICD.csv](./CSE6250-Project/data/codes/PROCEDURES_ICD.csv)
  * [lookups/](./CSE6250-Project/data/lookups)
  * [model/](./CSE6250-Project/data/model)
  * [notes/](./CSE6250-Project/data/notes)
    * [NOTEEVENTS.csv](./CSE6250-Project/data/notes/NOTEEVENTS.csv)
* [.gitignore](./CSE6250-Project/.gitignore)
* [README.md](./CSE6250-Project/README.md)
* [config.yml](./CSE6250-Project/config.yml)
* [environment.yml](./CSE6250-Project/environment.yml)
* [preprocess.py](./CSE6250-Project/preprocess.py)
* [train.py](./CSE6250-Project/train.py)



## Prepare Dataset

```python preprocess.py```

This will take a few minutes to run. The prepared dataset will be saved as pickle files in `data/model/`.

## Train Model

**IN PROGRESS**