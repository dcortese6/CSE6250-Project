#!/usr/bin/env python
import argparse
import pickle
import torch
import yaml

import numpy as np

np.random.seed(0)
from models import Model

import sys
sys.dont_write_bytecode = True
import configparser, os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from i2b2_preprocess import DatasetProvider
import i2b2

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass

import warnings
warnings.warn = warn

NGRAM_RANGE = (1, 1) # use unigrams for cuis
MIN_DF = 0.0

global args

parser = argparse.ArgumentParser(description='SVM')
parser.add_argument('--config', default='config.yml')
args, unknown = parser.parse_known_args()

with open(args.config, 'r') as f:
  configs = yaml.safe_load(f)
  f.close()

judgement = configs['eval']['judgement']
train_data = configs['eval']['train_data']
train_annot = configs['eval']['train_annot']
test_data = configs['eval']['test_data']
test_annot = configs['eval']['test_annot']
min_token_freq = configs['eval']['min_token_freq']

model_dict = torch.load("./output/models/BaseModel.pth")
max_token = model_dict['embed.weight'].shape[0]
model = Model(embeddings=max_token)

model.load_state_dict(model_dict)
for param in model.parameters():
    param.requires_grad = False

layers = list(model.children())

def run_evaluation_sparse(disease, judgement):
  """Train on train set and evaluate on test set"""

  print('disease:', disease)
  print('judgement:', judgement)

  # handle training data first -- think we create alphabet pickle here first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease, # i2b2.get_disease_names(test_annot, exclude):
    judgement, # INTUITIVE
    use_pickled_alphabet=False,
    alphabet_pickle=configs['eval']['alphabet_pickle'])
  x_train, y_train = train_data_provider.load_raw() # ISSUE?

  # VECTORIZE TRAINING DATA W/ COUNTING
  vectorizer = CountVectorizer(
    ngram_range=NGRAM_RANGE,
    stop_words='english',
    min_df=MIN_DF,
    vocabulary=None,
    binary=False)
  train_count_matrix = vectorizer.fit_transform(x_train)

  tf = TfidfTransformer()
  train_tfidf_matrix = tf.fit_transform(train_count_matrix)
  print("Shape of train_tfidf_matrix is: "+ str(train_tfidf_matrix.shape))

  # now handle the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=configs['eval']['alphabet_pickle'])
  x_test, y_test = test_data_provider.load_raw()
  print('test examples:', len(x_test))

  test_count_matrix = vectorizer.transform(x_test)
  test_tfidf_matrix = tf.transform(test_count_matrix)

  classifier = LinearSVC(class_weight='balanced')
  classifier.fit(train_tfidf_matrix, y_train)
  predictions = classifier.predict(test_tfidf_matrix)

  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print('unique labels in train:', len(set(y_train)))
  print('p = %.3f' % p)
  print('r = %.3f' % r)
  print('f1 = %.3f\n' % f1)

  return p, r, f1

def run_evaluation_svd(disease, judgement):
  """Train on train set and evaluate on test set"""

  print('disease:', disease)
  print('judgement:', judgement)


  # handle training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    use_pickled_alphabet=False,
    alphabet_pickle=configs['eval']['alphabet_pickle'])
  x_train, y_train = train_data_provider.load_raw()
  print('train examples:', len(x_train))

  vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
  train_tfidf_matrix = vectorizer.transform(x_train)

  # now handle the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=configs['eval']['alphabet_pickle'])
  x_test, y_test = test_data_provider.load_raw()
  print('test examples:', len(x_test))
  test_tfidf_matrix = vectorizer.transform(x_test)

  # load svd model and map train/test to low dimensions
  print('input shape:', train_tfidf_matrix.shape)
  svd = pickle.load(open('svd.pkl', 'rb'))
  train_tfidf_matrix = svd.transform(train_tfidf_matrix)
  test_tfidf_matrix = svd.transform(test_tfidf_matrix)
  print('output shape:', train_tfidf_matrix.shape)

  classifier = LinearSVC(class_weight='balanced')
  classifier.fit(train_tfidf_matrix, y_train)
  predictions = classifier.predict(test_tfidf_matrix)

  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print('unique labels in train:', len(set(y_train)))
  print('p = %.3f' % p)
  print('r = %.3f' % r)
  print('f1 = %.3f\n' % f1)

  print('%.3f & %.3f & %.3f\n' % (p, r, f1))

  return p, r, f1

def dense_model(input):
  x = layers[0](input)
  x = x.mean(dim=1)
  x = layers[1](x)
  return x

# HAVE TO USE ALPHABET.TXT FROM Neural Network Model -- TO LOAD UP THE MODEL, ITLL HAVE TO MATCH OUR EMBEDDINGS
def run_evaluation_dense(disease, judgement):
  # Use pre-trained patient representations

  print("disease: ", disease)
  print("judgement ", judgement)

  # load training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=configs['eval']['alphabet_pickle_learned_model'],
    min_token_freq=min_token_freq
  )
  x_train, y_train = train_data_provider.devin_load()
  print(x_train)

  print('train examples:', len(x_train))
  classes = len(set(y_train))
  print('unique labels in train:', classes)
  maxlen = len(max(x_train, key=len))
  print('padding train sequences to length:', maxlen)
  print(x_train)
  x_train = np.array([i + [0] * (maxlen - len(i)) for i in x_train])
  print('original x_train shape:', x_train.shape)
  print(x_train)
  x_train = dense_model(torch.from_numpy(x_train)).detach().numpy()
  print('new x_train shape:', x_train.shape)

  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=configs['eval']['alphabet_pickle_learned_model'],
    min_token_freq=min_token_freq
  )
  x_test, y_test = test_data_provider.devin_load()
  print('test examples:', len(x_test))

  classes = len(set(y_test))
  print('unique labels in test:', classes)
  maxlen = len(max(x_test, key=len))
  print('padding test sequences to length:', maxlen)
  x_test = np.array([i + [0] * (maxlen - len(i)) for i in x_test])

  print('original x_test shape:', x_test.shape)
  x_test = dense_model(torch.from_numpy(x_test)).detach().numpy()
  print('new x_test shape:', x_test.shape)

  classifier = LinearSVC(class_weight='balanced')
  classifier.fit(x_train, y_train)
  predictions = classifier.predict(x_test)
  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print('p = %.3f' % p)
  print('r = %.3f' % r)
  print('f1 = %.3f\n' % f1)

  return p, r, f1

def run_evaluation_all_diseases():
  """Evaluate classifier performance for all 16 comorbidities"""

  exclude = set()

  cfg = configparser.ConfigParser()
  cfg.read('dense.cfg')

  judgement = cfg.get('data', 'judgement')
  test_annot = cfg.get('data', 'test_annot')
  ps = []
  rs = []
  f1s = []
  # Bag-Of-Token Sparse Representation
  for disease in i2b2.get_disease_names(test_annot, exclude):
      p, r, f1 = run_evaluation_sparse(disease, judgement)
      ps.append(p)
      rs.append(r)
      f1s.append(f1)

  print("Sparse Representation")
  print('average p =', np.mean(ps))
  print('average r =', np.mean(rs))
  print('average f1 =', np.mean(f1s))

  ps = []
  rs = []
  f1s = []
  # Use SVD Patient-Token MIMIC Representation
  for disease in i2b2.get_disease_names(test_annot, exclude):
      p, r, f1 = run_evaluation_svd(disease, judgement)
      ps.append(p)
      rs.append(r)
      f1s.append(f1)

  print("SVD MIMIC Representation")
  print('average p =', np.mean(ps))
  print('average r =', np.mean(rs))
  print('average f1 =', np.mean(f1s))

  ps = []
  rs = []
  f1s = []
  # Use Learned Patient Representation
  for disease in i2b2.get_disease_names(test_annot, exclude):
      p, r, f1 = run_evaluation_dense(disease, judgement)
      ps.append(p)
      rs.append(r)
      f1s.append(f1)

  print("Learned Representation")
  print('average p =', np.mean(ps))
  print('average r =', np.mean(rs))
  print('average f1 =', np.mean(f1s))

if __name__ == "__main__":
  run_evaluation_all_diseases()
