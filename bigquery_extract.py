from google.cloud import bigquery

def connect(project='project_id'):
    client = bigquery.Client(project=project)
    return client

def get_notes(client):
    frame = client.query("""
        SELECT SUBJECT_ID, TEXT
        FROM `physionet-data.mimiciii_notes.noteevents`
    """).to_dataframe()
    return frame

def get_diagnoses(client):
    frame = client.query("""
        SELECT SUBJECT_ID, ICD9_CODE
        FROM `physionet-data.mimiciii_clinical.diagnoses_icd`
    """).to_dataframe()
    return frame

def get_procedures(client):
    frame = client.query("""
        SELECT SUBJECT_ID, ICD9_CODE
        FROM `physionet-data.mimiciii_clinical.procedures_icd`
    """).to_dataframe()
    return frame

def get_cpt(client):
    frame = client.query("""
        SELECT SUBJECT_ID, CPT_NUMBER
        FROM `physionet-data.mimiciii_clinical.cptevents`
    """).to_dataframe()
    return frame