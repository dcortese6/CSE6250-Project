import yaml
import argparse

import torch
import torch.nn as nn

from gensim.models import KeyedVectors

global args
parser = argparse.ArgumentParser(description='Patient Representation')
parser.add_argument('--config', default='config.yml')
args = parser.parse_args()

#Load yaml configs into configs dictionary
with open(args.config,'r') as f:
    configs = yaml.safe_load(f)
    f.close()

class Model(nn.Module):
    def __init__(self, embeddings):
        super(Model, self).__init__()
        if configs['data']['w2v']:
            self.init_vectors = KeyedVectors.load("./output/embeddings/word2vec.wordvectors", mmap='r')
            self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(self.init_vectors.vectors), freeze=False)
        else:
            self.embed = nn.Embedding(num_embeddings=embeddings,
                                    embedding_dim=configs['dan']['embdims'],
                                    padding_idx=0)
        self.hidden = nn.Linear(configs['dan']['embdims'], configs['dan']['hidden'])
        self.fc = nn.Linear(configs['dan']['hidden'], 174)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1)
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    
    

class BiGRU(nn.Module):
    def __init__(self, embeddings):
        super(BiGRU, self).__init__()
        if configs['data']['w2v']:
            self.init_vectors = KeyedVectors.load("./output/embeddings/word2vec.wordvectors", mmap='r')
            self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(self.init_vectors.vectors), freeze=False)
        else:
            self.embed = nn.Embedding(num_embeddings=embeddings,
                                    embedding_dim=configs['dan']['embdims'],
                                    padding_idx=0)
        # self.hidden = nn.Linear(configs['dan']['embdims'], configs['dan']['hidden'])
        # self.fc = nn.Linear(configs['dan']['hidden'], 174)
        self.gru = nn.GRU(input_size=configs['dan']['embdims'], 
                        hidden_size=configs['dan']['hidden'],
                        num_layers=2, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(in_channels=configs['dan']['hidden'],
                                out_channels=configs['dan']['hidden'],
                                kernel_size=2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(configs['dan']['hidden'], 174)
        
        
    def forward(self, x):
        x = self.embed(x)
        x, h = self.gru(x)
        x = self.conv1d(h.transpose(1,2).transpose(0,2))
        x = self.pool(x).squeeze()
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    