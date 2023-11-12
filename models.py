import yaml
import argparse

import torch
import torch.nn as nn

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
    