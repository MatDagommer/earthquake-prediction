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
from train_model import train_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-model","--model", type=int, help="choose model: 0 for CNN 1D, 1 for LSTM.", required = True)
parser.add_argument("-n", "--n_rows", type = int, help = "number of rows for training", default = 200_000_000)
parser.add_argument("-vr", "--valid_ratio", type = float, help="validation ratio. Default: 0.1", default = 0.1)
parser.add_argument("-lr", "--learning_rate", type = float, help="learning rate. Default: 0.0001", default = 0.0001)
parser.add_argument("-ep", "--num_epochs", type=int, help="number of epochs. Default: 500", default = 100)
parser.add_argument("-bs", "--batch_size", type=int, help="batch size or number of samples per batch. Default: 100", default = 100)
parser.add_argument("-o", "--overlap_rate", type = float, help="overlap rate. Default = 0.2", default = 0.2)
parser.add_argument("-s", "--seed", type=int, help = "seed for random numbers", default = 0)
parser.add_argument("-nsp", "--N_subpatches", type=int, help = "number of subpatches for lstm engineered sequences", default = 250)
args = parser.parse_args()


seed = args.seed
th.manual_seed(seed)
np.random.seed(seed)

### Ensure CPU is connected

print("\nCuda available: ", th.cuda.is_available())
print("Device Name: ", th.cuda.get_device_name(), "\n")
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# Model

if args.model not in [0, 1]:
    raise Exception("model_id must be either 0 (CNN 1D) or 1 (LSTM).")
else:
    if args.model == 0:
        model_id = "CNN"
    else:
        model_id = "LSTM"


nrows = args.n_rows
valid_ratio = args.valid_ratio
learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size # number of samples per batch
overlap_rate = args.overlap_rate
N_subpatches = args.N_subpatches


print("Training Model with Parameters:\n\nLearning Rate: {}\nEpochs: {}\nBatch Size: {}\nOverlap Rate: {}\nValidation Ratio: {}\n".format(learning_rate, num_epochs, batch_size, overlap_rate, valid_ratio))


### Load Data

rootpath = os.getcwd() + "/"

print("Loading Data...")
print("Loading Data: ", end="")
train_data = pd.read_csv(rootpath + "train.csv", usecols = ['acoustic_data', 'time_to_failure'], \
                         dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float64}, nrows = nrows)
print("Done.\n")

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


### Train Model

if model_id == "LSTM":
    print("Engineering Input Data...")
    print("Engineering Input Data: ", end = "")
    X_train, Y_train, X_valid, Y_valid = \
        lstm_feature_engineering(X_train, Y_train, X_valid, Y_valid, patch_size, N_sub_patches = N_subpatches)
    print("Done.\n")

N_samples = X_train.shape[0]
n_batch = int(X_train.shape[0] / batch_size)

print("Training the model..\n")
model, train_losses, valid_losses, valid_differences, best_mvd, best_mtd, min_tl, min_vl = \
    train_model(n_batch, batch_size, patch_size, num_epochs, learning_rate, model_id, X_train, Y_train, X_valid, Y_valid)


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