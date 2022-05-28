from re import I
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt 

import torch as th
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
th.cuda.empty_cache()

from torchsummary import summary 

import gzip
import pickle
import time

import pandas as pd
import scipy
import scipy.stats as stats
from scipy.stats import kurtosis, skew
import csv
import os
import pickle
import gc

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sequence_patches import sequence_patches
from plot_results import plot_results
from Cnn1d import Cnn1d
from check_accuracy import check_accuracy
from Lstm import Lstm
from lstm_feature_engineering import lstm_feature_engineering

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=int, help="choose model: 0 for CNN 1D, 1 for LSTM.")
parser.add_argument("-n", "--n_rows", help = "number of rows for training", default = 200_000_000)
parser.add_argument("-vr", "--valid_ratio", help="validation ratio. Default: 0.1", default = 0.1)
parser.add_argument("-lr", "--learning_rate", help="learning rate. Default: 0.0001", default = 0.0001)
parser.add_argument("-ep", "--num_epochs", help="number of epochs. Default: 500", default = 100)
parser.add_argument("-bs", "--batch_size", help="batch size or number of samples per batch. Default: 100", default = 100)
parser.add_argument("-o", "--overlap_rate", help="overlap rate. Default = 0.2", default = 0.2)
parser.add_argument("-s", "--seed", help = "seed for random numbers", default = 0)
args = parser.parse_args()


seed = args.seed
th.manual_seed(seed)
np.random.seed(seed)


# Model

if args.model_id not in [0, 1]:
    raise Exception("model_id must be either 0 (CNN 1D) or 1 (LSTM).")
else:
    if args.model_id == 0:
        model_id = "CNN"
    else:
        model_id = "LSTM"


nrows = args.n_rows
valid_ratio = args.valid_ratio
learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size # number of samples per batch
overlap_rate = args.overlap_rate


print("\nTraining Model with Parameters:\n\nLearning Rate: {}\nEpochs: {}\nBatch Size: {}\nOverlap Rate: {}\nValidation Ratio: {}\n".format(learning_rate, num_epochs, batch_size, overlap_rate, valid_ratio))

### Ensure CPU is connected

print("Cuda available: ", th.cuda.is_available())
print("Device Name: ", th.cuda.get_device_name())
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

### Load Data

rootpath = os.getcwd() + "/"

print("Loading Data: ", end="")
train_data = pd.read_csv(rootpath + "train.csv", usecols = ['acoustic_data', 'time_to_failure'], \
                         dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float64}, nrows = nrows)
print("Done.")

X = th.squeeze(th.tensor(train_data['acoustic_data'].values, dtype = th.int16))
Y = th.squeeze(th.tensor(train_data['time_to_failure'].values, dtype = th.float64))

del train_data; gc.collect()


### Retrieve Test Data

path = rootpath + "test/" + "seg_00030f.csv"
test = pd.read_csv(path)
patch_size = test.shape[0] # Defining patch size as the length of test sequences


### Normalize Data

ss = StandardScaler()
mm = MinMaxScaler()
    
X = th.squeeze(th.from_numpy(ss.fit_transform(np.array(X).reshape(-1, 1))))
Y = th.squeeze(th.from_numpy(mm.fit_transform(np.array(Y).reshape(-1, 1))))


### Generate small patches of the (otherwise massive) training sequence

_, X_patch, Y_patch = sequence_patches(patch_size, X, Y, overlap_rate = overlap_rate)

del X; del Y; gc.collect()

print(X_patch.shape)
print("There are ", X_patch.shape[0], "time series available for training and validation after patching with overlap. \n")


### Initializating Data Sets

# shuffle data
idx = np.arange(X_patch.shape[0]) # indices
shuffled_idx = np.random.shuffle(idx)
X_patch = th.squeeze(X_patch[shuffled_idx,:])
Y_patch = th.squeeze(Y_patch[shuffled_idx])

N_samples = X_patch.shape[0]
N_valid = int(valid_ratio*N_samples) # number of validation patches

X_valid = X_patch[:N_valid,:].cpu()
Y_valid = Y_patch[:N_valid].cpu()

# Inputs of nn.Conv1d must have the following shape: (N, C_in, *),
#where N: number of samples (batch size), C_in: number of input channels, *: can be any dimension (150_000 for our time series)

# Add channel dimension
X_valid = th.unsqueeze(X_valid, 1)

X_train = X_patch[N_valid:,:].cpu()
Y_train = Y_patch[N_valid:].cpu()
N_train = X_train.shape[0]
X_train = X_train.reshape([N_train, 1, patch_size])


### Initialize Network, Define Loss and optimizer

if model_id == "CNN": # CNN 1D

    n_batch = int(X_train.shape[0] / batch_size) # total number of batches

    model = Cnn1d()
    model.to(device)

elif model_id == "LSTM": # LSTM

    # Hyperparameters
    hidden_size = 1 #number of features in hidden state
    num_layers = 1 #number of stacked lstm layers => should stay at 1 unless you want to combine two LSTMs together
    N_sub_patches = 250

    X_train_, Y_train_, X_valid_, Y_valid_ = \
        lstm_feature_engineering(X_train, Y_train, X_valid, Y_valid, patch_size, N_sub_patches = N_sub_patches)
    N_features = X_train_.shape[-1]
    L_seq = X_train_.shape[2]
    model = Lstm(N_features, hidden_size, num_layers, L_seq) #our lstm class
    

loss = th.nn.MSELoss()    # mean-squared error for regression
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate) 



### Train network

th.cuda.empty_cache()
if model_id == "CNN": # CNN 1D

    train_losses, valid_losses = [], []

    best_mvd = [] # valid differences at best epoch
    best_mtd = [] # training differences at best epoch
    min_vl = 1000
    min_tl = 1000

    N_train_per_epoch = n_batch*batch_size 

    start = time.perf_counter()

    for epoch in range(num_epochs):

        th.cuda.empty_cache()
        
        # Generating Random Batches every epoch
        random_idx = np.random.randint(0, N_train, N_train_per_epoch)
        X_train_batch = th.reshape(X_train[random_idx], [n_batch, batch_size, 1, patch_size])
        Y_train_batch = th.reshape(Y_train[random_idx], [n_batch, batch_size, 1, 1])
        
        running_loss = 0
        _loss = 0

        for ids in range (n_batch):
            
            # Setting gradients to zero
            optimizer.zero_grad()
            # Model prediction
            predicted_ttfs = th.squeeze(model(X_train_batch[ids].to(device))).cpu()
            # Loss function
            _loss = loss(predicted_ttfs, th.squeeze(Y_train_batch[ids]))
            
            del predicted_ttfs
            gc.collect()
            
            # Backpropagation
            _loss.backward()
            running_loss += _loss.item()
            
            # Updating model weights
            optimizer.step()
            
        optimizer.zero_grad()
        th.cuda.empty_cache()
        
        # Turning evaluation mode ON (No Dropout / BatchNorm layers)
        model.eval()

        with th.no_grad():
            valid_loss, valid_differences = check_accuracy(X_valid, Y_valid, model)
            valid_losses.append(valid_loss)
            train_losses.append(running_loss / n_batch)
            
            if valid_loss < min_vl:
                
                min_vl = valid_loss
                min_tl = running_loss
                
                best_mvd = valid_differences
                
                Y_final = th.squeeze(model(X_train[:int(N_train/10)].to(device))).cpu()
                
                best_mtd = Y_train[:int(N_train/10)] - Y_final[:]
                best_mtd = best_mtd.cpu().detach().numpy()
        
        # Turning training mode back on
        model.train()
        
        if epoch%10 == 0:
            print("Epoch: {}\t".format(epoch),
                    "train Loss: {:.5f}.. ".format(train_losses[-1]),
                    "valid Loss: {:.5f}.. ".format(valid_losses[-1])) 

    print("---------- Best : {:.3f}".format(min(valid_losses)), " at epoch " 
        , np.fromiter(valid_losses, dtype=float).argmin(), " / ", epoch + 1)



elif model_id == "LSTM": # LSTM

    # Training LSTM

    valid_losses = np.zeros((num_epochs))
    train_losses = np.zeros((num_epochs))

    min_vl = 1000

    best_mvd = []
    best_mtd = []

    for epoch in range(num_epochs):
        outputs = model.forward(X_train_) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss(outputs, Y_train_)
    
        loss.backward() #calculates the loss of the loss function

        train_loss = loss.item()
        train_losses[epoch] = train_loss

        with th.no_grad():
            
            valid_output = model(X_valid_)
            valid_loss = loss(valid_output, Y_valid_)
            valid_losses[epoch] = valid_loss.item()
            valid_differences = valid_output - Y_valid_

            if valid_loss < min_vl:

                min_vl = valid_loss
                min_tl = train_loss

                best_mvd = valid_differences

                Y_final = model(X_train_)

                best_mtd = Y_train_ - Y_final
                best_mtd = best_mtd.numpy()
            
    
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 


end = time.perf_counter()
print("\ntime elapsed: {:.3f}\n".format(end - start))


model_name = input("Choose a name for the model: ")
os.mkdir('Models/' + model_name)

# Plot Results
plot_results(train_losses, valid_losses, best_mvd, best_mtd, min_tl, min_vl, mm, model_name)


# Saving main model features for later retrieval 
model_features = {"N_samples": N_samples, "N_train": N_train, "N_valid": N_valid, \
                    "overlap_rate": overlap_rate, "learning_rate": learning_rate, "num_epochs": num_epochs, \
                    "seed": seed, "batch_size": batch_size, "train_losses": train_losses, "valid_losses": valid_losses, \
                    "valid_differences": valid_differences, "best_mtd": best_mtd, "best_mvd": best_mvd, "min_tl": min_tl, \
                    "min_vl": min_vl}

pickle.dump(model_features, open('Models/' + model_name + '/' + model_name + ".p", "wb" ))
    
model_features_display = {"N_samples": N_samples, "N_train": N_train, "N_valid": N_valid, \
                  "overlap_rate": overlap_rate, "learning_rate": learning_rate, "num_epochs": num_epochs, \
                 "seed": seed, "batch_size": batch_size}
    
pickle.dump(model_features_display, open('Models/' + model_name + '/' + model_name + "_display.p", "wb" ))