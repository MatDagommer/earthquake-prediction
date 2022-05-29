# LSTM Feature Engineering

import torch as th
import numpy as np 
import scipy

from feature_expansion import feature_expansion

def lstm_feature_engineering(X_train, Y_train, X_valid, Y_valid, patch_size, N_sub_patches):

    X_train = np.array(th.squeeze(X_train)); Y_train = np.array(Y_train)
    X_valid = np.array(th.squeeze(X_valid)); Y_valid = np.array(Y_valid)

    L_seq = X_train.shape[1]
    L_sub_patch = int(L_seq / N_sub_patches)
    N_features = 10 # mean, std, skew, kurt, min, max, quantiles 0.25, 0.5, 0.75, inter-quartile range
    
    X_train_rearranged = feature_expansion(X_train, N_sub_patches, L_sub_patch)
    X_valid_rearranged = feature_expansion(X_valid, N_sub_patches, L_sub_patch)

    X_train = th.from_numpy(X_train_rearranged)
    X_valid = th.from_numpy(X_valid_rearranged)

    Y_train = th.from_numpy(Y_train); Y_valid = th.from_numpy(Y_valid)
    #Y_train = th.unsqueeze(Y_train, -1); Y_valid = th.unsqueeze(Y_valid, -1)

    return X_train, Y_train, X_valid, Y_valid