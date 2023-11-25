import os
import yaml
import argparse
import utils
import torch
import pickle
import numpy as np
import torch.nn as nn

from models import Model
import eval_utils
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

global args
parser = argparse.ArgumentParser(description='SVM')
parser.add_argument('--config', default='config.yml')
args, unknown = parser.parse_known_args()

# with open(args.config,'r') as f:
#     configs = yaml.safe_load(f)
#     f.close()
    
# big_query = configs['data']['bq']

# if not big_query:
#     with open('data/model/model_data_x.pkl', 'rb') as f:
#         x = pickle.load(f)

#     with open('data/model/model_data_y.pkl', 'rb') as f:
#         y = pickle.load(f)

with open(args.config,'r') as f:
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
layers = list(model.children())

def dense_model(input):
    x = layers[0](input)
    x = x.mean(dim=1)
    x = layers[1](x)
    return x

def run_evaluation_dense(disease, judgement):
    
    # load training data first
    train_data_provider = eval_utils.DatasetProvider(
        train_data,
        train_annot,
        disease,
        judgement,
        # use_pickled_alphabet=True,
        # alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
        min_token_freq=min_token_freq
    )
    x_train, y_train = train_data_provider.load()
    
    classes = len(set(y_train))
    print('unique labels in train:', classes)
    # maxlen = cfg.getint('data', 'maxlen')
    maxlen = len(max(x_train, key=len))
    print('padding train sequences to length:', maxlen)
    # x_train = pad_sequences(x_train, maxlen=maxlen)
    x_train = np.array([i + [0]*(maxlen-len(i)) for i in x_train])
    
    print('original x_train shape:', x_train.shape)
    x_train = dense_model(torch.from_numpy(x_train)).detach().numpy()
    print('new x_train shape:', x_train.shape)
    
    test_data_provider = eval_utils.DatasetProvider(
        test_data,
        test_annot,
        disease,
        judgement,
        # use_pickled_alphabet=True,
        # alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
        min_token_freq=min_token_freq
    )
    x_test, y_test = test_data_provider.load()
    
    classes = len(set(y_test))
    print('unique labels in test:', classes)
    # maxlen = cfg.getint('data', 'maxlen')
    maxlen = len(max(x_test, key=len))
    print('padding test sequences to length:', maxlen)
    # x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = np.array([i + [0]*(maxlen-len(i)) for i in x_test])
    
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
    


ps = []
rs = []
f1s = []

exclude = set()
for disease in eval_utils.get_disease_names(test_annot, exclude):
    print("Running dense evaluation on disease:", disease)
    p, r, f1 = run_evaluation_dense(disease, judgement)
    # print(disease)
    ps.append(p)
    rs.append(r)
    f1s.append(f1)
    
print('average p =', np.mean(ps))
print('average r =', np.mean(rs))
print('average f1 =', np.mean(f1s))