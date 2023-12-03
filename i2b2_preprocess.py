#!/usr/bin/env python

import sys

from nltk.corpus import stopwords

sys.dont_write_bytecode = True
sys.path.append('../Lib/')
import xml.etree.ElementTree as et

import utils, i2b2
import configparser, os, pickle
import collections

LABEL2INT = {'Y':0, 'N':1, 'Q':2, 'U':3}

# file to log alphabet entries for debugging
ALPHABET_FILE = 'i2b2/Model/alphabet.txt'

class DatasetProvider:
  """Comorboditiy data loader"""

  def __init__(self,
               corpus_path,
               annot_xml,
               disease,
               judgement,
               use_pickled_alphabet=False,
               alphabet_pickle=None,
               min_token_freq=0,
               max_token_freq=0):
    """Index words by frequency in a file"""

    self.corpus_path = corpus_path
    self.annot_xml = annot_xml
    self.min_token_freq = min_token_freq
    self.max_token_freq = max_token_freq
    self.disease = disease
    self.judgement = judgement
    self.alphabet_pickle = alphabet_pickle

    self.token2int = {}

    # when training, make alphabet and pickle it
    # when testing, load it from pickle
    if use_pickled_alphabet:
      print('reading alphabet from', alphabet_pickle)
      pkl = open(alphabet_pickle, 'rb')
      self.token2int = pickle.load(pkl)
    else:
      print('getting tokens and counts and dumping them to file...')
      self.make_token_alphabet()

  def extract_tokens(self, text):
    text = text.lower()

    tokens = []
    for token in text.split():
      if token.isalpha():
        tokens.append(token)
    if len(tokens) > self.max_token_freq:
      return None
    return tokens

  def make_token_alphabet(self):
    """Map tokens (CUIs) to integers"""

    # count tokens in the entire corpus
    token_counts = collections.Counter()

    for f in os.listdir(self.corpus_path):
      file_path = os.path.join(self.corpus_path, f)

      text = open(file_path).read()
      file_feat_list = self.extract_tokens(text)

      if file_feat_list is None:
        continue

      token_counts.update(file_feat_list)

    # now make alphabet (high freq tokens first)
    index = 1
    self.token2int['oov_word'] = 0
    outfile = open(ALPHABET_FILE, 'w')

    stop_words = stopwords.words('english')
    for token, count in token_counts.most_common():
      if count > self.min_token_freq and token not in stop_words:

        outfile.write('%s|%s\n' % (token, count))
        self.token2int[token] = index
        index = index + 1

    outfile.close()
    # pickle alphabet
    pickle_file = open(self.alphabet_pickle, 'wb')
    pickle.dump(self.token2int, pickle_file)
    pickle_file.close()

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices for keras"""

    labels = []    # int labels
    examples = []  # examples as int sequences
    no_labels = [] # docs with no labels

    # document id -> label mapping
    doc2label = i2b2.parse_standoff(
      self.annot_xml,
      self.disease,
      self.judgement)

    # load examples and labels
    for f in os.listdir(self.corpus_path):
      doc_id = f.split('.')[0]

      file_path = os.path.join(self.corpus_path, f)
      text = open(file_path).read()
      file_feat_list = self.extract_tokens(text)

      if file_feat_list is None:
        continue

      example = []
      for token in set(file_feat_list):
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]

      # no labels for some documents for some reason
      if doc_id in doc2label:
        string_label = doc2label[doc_id]
        int_label = LABEL2INT[string_label]
        labels.append(int_label)
        examples.append(example)
      else:
        no_labels.append(doc_id)

    print('%d documents with no labels for %s/%s in %s' \
      % (len(no_labels), self.disease,
         self.judgement, self.annot_xml.split('/')[-1]))

    return examples, labels

  def devin_load(self, maxlen=float('inf')):
    """Load for sklearn training"""

    labels = []    # string labels
    examples = []  # examples as strings
    no_labels = [] # docs with no labels

    # document id -> label mapping
    doc2label = i2b2.parse_standoff(
      self.annot_xml,
      self.disease,
      self.judgement)

    for f in os.listdir(self.corpus_path):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)
      example = []

      for token in set(file_feat_list):
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]

      # no labels for some documents for some reason
      if doc_id in doc2label:
        string_label = doc2label[doc_id]
        int_label = LABEL2INT[string_label]
        labels.append(int_label)
        examples.append(example)
      else:
        no_labels.append(doc_id)

    print('%d documents with no labels for %s/%s in %s' \
      % (len(no_labels), self.disease,
         self.judgement, self.annot_xml.split('/')[-1]))

    return examples, labels

  def load_raw(self):
    """Load for sklearn training"""

    labels = []    # string labels
    examples = []  # examples as strings
    no_labels = [] # docs with no labels

    # document id -> label mapping
    doc2label = i2b2.parse_standoff(
      self.annot_xml,
      self.disease,
      self.judgement)

    for f in os.listdir(self.corpus_path):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.corpus_path, f)
      file_feat_list = utils.read_cuis(file_path)

      # no labels for some documents for some reason
      if doc_id in doc2label:
        string_label = doc2label[doc_id]
        int_label = LABEL2INT[string_label]
        labels.append(int_label)
        examples.append(' '.join(file_feat_list))

      else:
        no_labels.append(doc_id)

    print('%d documents with no labels for %s/%s in %s' \
      % (len(no_labels), self.disease,
         self.judgement, self.annot_xml.split('/')[-1]))

    return examples, labels

if __name__ == "__main__":
  cfg = configparser.ConfigParser()
  cfg.read('sparse.cfg')

  data_dir = os.path.join(cfg.get('data', 'train_data'))
  annot_xml = os.path.join(cfg.get('data', 'train_annot'))

  dataset = DatasetProvider(data_dir, annot_xml, disease="asthma", judgement=cfg.get('data', 'judgement'), alphabet_pickle ='i2b2/Model/alphabet.pkl')
  dataset.make_token_alphabet()
