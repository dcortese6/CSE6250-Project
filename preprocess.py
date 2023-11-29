import os
import yaml
import argparse
import collections
import pickle 
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import word2vec

global args
parser = argparse.ArgumentParser(description='Patient Representation')
parser.add_argument('--config', default='config.yml')
args = parser.parse_args()

#Load yaml configs into configs dictionary
with open(args.config,'r') as f:
    configs = yaml.safe_load(f)
    f.close()
    
ALPHABET_FILE = 'data/lookups/alphabet.txt'
ALPHABET_PICKLE = 'data/lookups/alphabet.pkl'
CODE_FREQ_FILE = 'data/lookups/codes.txt'
NOTES_FILE = 'NOTEEVENTS.csv'
DIAG_ICD9_FILE = 'DIAGNOSES_ICD.csv'
PROC_ICD9_FILE = 'PROCEDURES_ICD.csv'
CPT_CODE_FILE = 'CPTEVENTS.csv'
MODEL_DATA_X = 'data/model/model_data_x.pkl'
MODEL_DATA_Y = 'data/model/model_data_y.pkl'
    
class DatasetProvider:
    def __init__(self,
            corpus_path,
            code_dir,
            min_token_freq,
            max_tokens_in_file,
            min_examples_per_code,
            use_bigquery,
            use_cuis=False):
        
        self.corpus_path = corpus_path
        self.code_dir = code_dir
        self.min_token_freq = min_token_freq
        self.max_tokens_in_file = max_tokens_in_file
        self.min_examples_per_code = min_examples_per_code
        self.use_cuis = use_cuis
        self.use_bigquery = use_bigquery
        
        self.token2int = {}  # words indexed by frequency
        self.code2int = {}   # class to int mapping
        self.subj2codes = {} # subj_id to set of icd9 codes
        
    # def read_cuis(self, file_name)   
    #     infile = os.path.join(self.corpus_path, file_name)
        if self.use_bigquery:
            import bigquery_extract as bq
            self.client = bq.connect()
            
        if not os.path.isfile(ALPHABET_PICKLE):
            print('cannot find tokens file. reading notes to create it...')
            if self.use_bigquery:
                notes = bq.get_notes(self.client)
            else:
                notes = pd.read_csv(os.path.join(self.corpus_path, NOTES_FILE), usecols=['SUBJECT_ID', 'TEXT'])
            notes = notes.groupby(['SUBJECT_ID'])['TEXT'].apply(lambda x: ' '.join(x)).reset_index()
            print('getting tokens and counts and dumping them to file...')
            self.make_and_write_token_alphabet(notes)
        print('retrieving alphabet from file...')
        self.token2int = pickle.load(open(ALPHABET_PICKLE, 'rb'))
        print('mapping codes...')
        diag_code_file = os.path.join(self.code_dir, DIAG_ICD9_FILE)
        proc_code_file = os.path.join(self.code_dir, PROC_ICD9_FILE)
        cpt_code_file = os.path.join(self.code_dir, CPT_CODE_FILE)
        self.map_subjects_to_codes(diag_code_file, 'ICD9_CODE', 'diag', 3)
        self.map_subjects_to_codes(proc_code_file, 'ICD9_CODE', 'proc', 2)
        self.map_subjects_to_codes(cpt_code_file, 'CPT_NUMBER', 'cpt', 5)
        self.make_code_alphabet()
            
    def extract_tokens(self, text):
        text = text.lower()
        
        tokens = []
        for token in text.split():
            if token.isalpha():
                tokens.append(token)
        if len(tokens) > self.max_tokens_in_file:
            return None
        return tokens
    
    def make_and_write_token_alphabet(self, file_name):
        token_counts = collections.Counter()
        texts = file_name['TEXT'].values
        for txt in texts:
            file_ngram_list = self.extract_tokens(txt)
            if file_ngram_list is None:
                continue
            token_counts.update(file_ngram_list)
        
        index = 1
        self.token2int['oov_word'] = 0
        outfile = open(ALPHABET_FILE, 'w')
        stop_words = stopwords.words('english')
        for token, count in token_counts.most_common():
            if count > self.min_token_freq and token not in stop_words:
                outfile.write('%s|%s\n' % (token, count))
                self.token2int[token] = index
                index += 1
        pickle_file = open(ALPHABET_PICKLE, 'wb')
        pickle.dump(self.token2int, pickle_file)
    
    
    def map_subjects_to_codes(self,
                            code_file,
                            code_col,
                            prefix,
                            num_digits):
        
        if self.use_bigquery:
            if prefix == 'diag':
                frame = bq.get_diagnoses(self.client)
            if prefix == 'proc':
                frame = bq.get_procedures(self.client)
            if prefix == 'cpt':
                frame = bq.get_cpt(self.client)
        else:
            frame = pd.read_csv(code_file, usecols=['SUBJECT_ID', code_col])
        for subj_id, code in zip(frame.SUBJECT_ID, frame[code_col]):
            if subj_id not in self.subj2codes:
                self.subj2codes[subj_id] = set()
            short_code = '%s_%s' % (prefix, str(code)[0:num_digits])
            self.subj2codes[subj_id].add(short_code)
    
    
    def make_code_alphabet(self):
        code_counter = collections.Counter()
        for codes in self.subj2codes.values():
            code_counter.update(codes)
        outfile = open(CODE_FREQ_FILE, 'w')
        for code, count in code_counter.most_common():
            outfile.write('%s|%s\n' % (code, count))
            
        index = 0
        for code, count in code_counter.most_common():
            if count > self.min_examples_per_code:
                self.code2int[code] = index
                index += 1
        
            
    def load(self,
            maxlen=float('inf'),
            tokens_as_set=True):
        
        codes = []
        examples = []
        word_vecs = []
        
        if self.use_bigquery:
            notes = bq.get_notes(self.client)
        else:
            notes = pd.read_csv(os.path.join(self.corpus_path, NOTES_FILE), usecols=['SUBJECT_ID', 'TEXT'])
        
        # notes = pd.read_csv(os.path.join(self.corpus_path, NOTES_FILE),
        #                     usecols=['SUBJECT_ID', 'TEXT'])
        notes = notes.groupby(['SUBJECT_ID'])['TEXT'].apply(lambda x: ' '.join(x)).reset_index()
        
        for subj_id, txt in zip(notes.SUBJECT_ID, notes.TEXT):
        # for txt in texts:
            file_ngram_list = self.extract_tokens(txt)
            if file_ngram_list is None:
                continue
            word_vecs.append(file_ngram_list)
            
            if len(self.subj2codes[subj_id]) == 0:
                print('skipping text for subject', subj_id)
                continue
            
            code_vec = [0] * len(self.code2int)
            for icd9 in self.subj2codes[subj_id]:
                if icd9 in self.code2int:
                    code_vec[self.code2int[icd9]] = 1
                    
            if sum(code_vec) == 0:
                continue
            
            codes.append(code_vec)
            
            example = []
            if tokens_as_set:
                file_ngram_list = set(file_ngram_list)
            for token in file_ngram_list:
                if token in self.token2int:
                    example.append(self.token2int[token])
                else:
                    example.append(self.token2int['oov_word'])
            if len(example) > maxlen:
                example = example[0:maxlen]
                
            examples.append(example)
        
            # init_vectors = None
        if configs['data']['w2v']:
            print('extracting word2vec embeddings....')
            # embed_file = os.path.join(base, cfg.get('data', 'embed'))
            embed_dir = './output/embeddings/word2vec.wordvectors'
            w2v = word2vec.Model()
            embeddings = w2v.get_embeddings(word_vecs, vector_size=configs['dan']['embdims'])
            if not os.path.exists(embed_dir):
                word_vectors = embeddings.wv
                word_vectors.save(embed_dir)
        
        return examples, codes
            
if __name__ == "__main__":

    big_query = configs['data']['bq']
    # use_w2v = configs['data']['w2v']
    # embed_dim = configs['dan']['embdims']
    # embed_dir = './output/embeddings/word2vec.wordvectors'

    dataset = DatasetProvider(
        configs['data']['notes'],
        configs['data']['codes'],
        configs['args']['min_token_freq'],
        configs['args']['max_tokens_in_file'],
        configs['args']['min_examples_per_code'],
        big_query
    )
    x, y = dataset.load()
    
    # init_vectors = None
    # if use_w2v:
    #     print('extracting word2vec embeddings....')
    #     # embed_file = os.path.join(base, cfg.get('data', 'embed'))
    #     w2v = word2vec.Model()
    #     embeddings = w2v.get_embeddings(x, vector_size=embed_dim)
    #     if not os.path.exists(embed_dir):
    #         word_vectors = embeddings.wv
    #         word_vectors.save(embed_dir)
        # init_vectors = [w2v.select_vectors(dataset.token2int)]
    
    print("padding dataset...")
    maxlen = len(max(x, key=len))
    x = np.array([i + [0]*(maxlen-len(i)) for i in x])
    
    if not big_query:
        pickle_x = open(MODEL_DATA_X, 'wb')
        pickle.dump(x, pickle_x)   
        pickle_y = open(MODEL_DATA_Y, 'wb')
        pickle.dump(y, pickle_y)   
