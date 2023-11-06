import os
import pickle
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import train, evaluate
from models import Model

global args
parser = argparse.ArgumentParser(description='Patient Representation')
parser.add_argument('--config', default='config.yml')
args = parser.parse_args()

#Load yaml configs into configs dictionary
with open(args.config,'r') as f:
    configs = yaml.safe_load(f)
    f.close()

with open('data/model/model_data_x.pkl', 'rb') as f:
    x = pickle.load(f)
    
with open('data/model/model_data_y.pkl', 'rb') as f:
    y = pickle.load(f)
    
PATH_OUTPUT = "./output/models/"
os.makedirs(PATH_OUTPUT, exist_ok=True)
    
NUM_EPOCHS = configs['dan']['epochs']
BATCH_SIZE = configs['dan']['batch']
NUM_WORKERS = configs['dan']['workers']
USE_CUDA = configs['dan']['cuda']

device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
if device.type == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
    
tensor_x = torch.tensor(x, dtype=torch.float32)
tensor_y = torch.tensor(y, dtype=torch.float32)    
    
train_x, test_x, train_y, test_y = train_test_split(tensor_x, tensor_y, test_size=configs['args']['test_size'])

# train_x = train_x[:1000]
# train_y = train_y[:1000]
# test_x = test_x[1000:1200]
# test_y = test_y[1000:1200]

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


model = Model()
save_file = 'Model.pth'

criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters())

model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

writer = SummaryWriter()
for epoch in range(NUM_EPOCHS):
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
    valid_loss, valid_accuracy = evaluate(model, device, test_loader, criterion)
    for m in train_accuracy:
        writer.add_scalars(m, {'train': train_accuracy[m],
                                'validation': valid_accuracy[m]}, epoch)
    writer.flush()
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

    is_best = valid_accuracy['macro/f1'] > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
    if is_best:
        best_val_acc = valid_accuracy['macro/f1']
        torch.save(model, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)

writer.close()
