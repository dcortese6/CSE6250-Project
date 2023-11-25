import os
import pickle
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

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
    
big_query = configs['data']['bq']

if not big_query:
    with open('data/model/model_data_x.pkl', 'rb') as f:
        x = pickle.load(f)

    with open('data/model/model_data_y.pkl', 'rb') as f:
        y = pickle.load(f)
    
MODEL_OUTPUT = "./output/models/"
PLOT_OUTPUT = "./output/plots/"
os.makedirs(MODEL_OUTPUT, exist_ok=True)
os.makedirs(PLOT_OUTPUT, exist_ok=True)
    
NUM_EPOCHS = configs['dan']['epochs']
BATCH_SIZE = configs['dan']['batch']
NUM_WORKERS = configs['dan']['workers']
USE_CUDA = configs['dan']['cuda']

device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
if device.type == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
    
tensor_x = torch.tensor(x, dtype=torch.int32)
tensor_y = torch.tensor(y, dtype=torch.float32) 
    
train_x, test_x, train_y, test_y = train_test_split(tensor_x, tensor_y, test_size=configs['args']['test_size'])


train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

max_token = np.max(x)
model = Model(embeddings=max_token+1)
model_name = 'BaseModel'

criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=configs['dan']['learnrt'])

model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

writer = SummaryWriter()
score_metric = configs['dan']['score']
for epoch in range(NUM_EPOCHS):
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch, score_metric)
    valid_loss, valid_accuracy = evaluate(model, device, test_loader, criterion, score_metric)
    # for m in train_accuracy:
        # print(m)
    writer.add_scalars(score_metric, {'train': train_accuracy,
                                    'validation': valid_accuracy}, epoch)
    writer.flush()
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

    is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
    if is_best:
        best_val_acc = valid_accuracy
        torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT, str(model_name) + ".pth"), _use_new_zipfile_serialization=False)

writer.close()

plt.plot(train_losses, label="Training Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend(loc='upper right')
plt.title("Loss Curve")
plt.savefig(PLOT_OUTPUT + model_name + "_Loss.png")
plt.close()

plt.plot(train_accuracies, label="Training Macro F1")
plt.plot(valid_accuracies, label="Validation Macro F1")
plt.xlabel("epoch")
plt.ylabel("Macro F1")
plt.legend(loc='upper left')
plt.title("Macro F1 Curve")
plt.savefig(PLOT_OUTPUT + model_name + "_F1.png")
