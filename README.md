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

## Download i2b2 Obesity Challenge Data

https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

Insert the following files into the directory `i2b2/Xml/`:

    - obesity_patient_records_test.xml
    - obesity_patient_records_training.xml
    - obesity_patient_records_training_2.xml
    - obesity_standoff_annotations_test.xml
    - obesity_standoff_annotations_training.xml
    - obesity_standoff_annotations_training_addendum3.xml


Be sure to have empty directories `data/lookups/`, `data/model/`, `i2b2/extracted_text/Test`, `i2b2/extracted_text/Train1+2`. Your folder structure should look like this before preparing the dataset:

* [data/](./CSE6250-Project/data)
  * [codes/](./CSE6250-Project/data/codes)
    * [CPTEVENTS.csv](./CSE6250-Project/data/codes/CPTEVENTS.csv)
    * [DIAGNOSES_ICD.csv](./CSE6250-Project/data/codes/DIAGNOSES_ICD.csv)
    * [PROCEDURES_ICD.csv](./CSE6250-Project/data/codes/PROCEDURES_ICD.csv)
  * [lookups/](./CSE6250-Project/data/lookups)
  * [model/](./CSE6250-Project/data/model)
  * [notes/](./CSE6250-Project/data/notes)
    * [NOTEEVENTS.csv](./CSE6250-Project/data/notes/NOTEEVENTS.csv)
* [i2b2/](./CSE6250-Project/i2b2)
  * [Xml/](./CSE6250-Project/i2b2/Xml)
    * [obesity_patient_records_test.xml](./CSE6250-Project/i2b2/Xml/obesity_patient_records_test.xml)
    * [obesity_patient_records_training.xml](./CSE6250-Project/i2b2/Xml/obesity_patient_records_training.xml)
    * [obesity_patient_records_training2.xml](./CSE6250-Project/i2b2/Xml/obesity_patient_records_training2.xml)
    * [obesity_standoff_annotations_test.xml](./CSE6250-Project/i2b2/Xml/obesity_standoff_annotations_test.xml)
    * [obesity_standoff_annotations_training.xml](./CSE6250-Project/i2b2/Xml/obesity_standoff_annotations_training.xml)
    * [obesity_standoff_annotations_training_addendum3.xml](./CSE6250-Project/i2b2/Xml/obesity_standoff_annotations_training_addendum3.xml)
  * [extracted_text/](./CSE6250-Project/i2b2/extracted_text)
    * [Test/](./CSE6250-Project/i2b2/extracted_text/Test)
    * [Train1+2/](./CSE6250-Project/i2b2/extracted_text/Train1+2)
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

Train a model by changing any desired parameters in config.yml and run:

```python train.py```

To view model progress, open another terminal start Tensorboard:

```tensorboard --logdir runs```

## Evaluate Model with i2b2 Obesity Challenge

Run to create training/test text files into `i2b2/extracted_text/` from the `i2b2/Xml/` annotation files.

```python i2b2.py```

Create pickled alphabet from the notes created in `i2b2/extracted_text/`.

```python i2b2_preprocess.py```

You will need to run this to be able to create the TF-IDF & SVD models from the MIMIC-III Patient-Tokens Matrix.

```python MIMIC_SVD_preprocess.py```

Lastly, the code below will run the needed models & metrics.

(1) Baseline SVM model with Bag of Tokens

(2) Baseline SVM model with a 1000-Dimensional Space SVD on a MIMIC-III Patient-Token Representation

(3) SVM Model with Learned Representation 

```python i2b2_svm.py```




