# LSTM Feature Engineering

import torch as th
import numpy as np 
import scipy

def lstm_feature_engineering(X_train, Y_train, X_valid, Y_valid, size_patch, N_sub_patches):

    X_train = np.array(th.squeeze(X_train)); Y_train = np.array(Y_train)
    X_valid = np.array(th.squeeze(X_valid)); Y_valid = np.array(Y_valid)

    L_seq = X_train.shape[1]
    L_sub_patch = int(L_seq / N_sub_patches)
    N_features = 10 # mean, std, skew, kurt, min, max, quantiles 0.25, 0.5, 0.75, inter-quartile range
  
    def rearranging_X(X):
        
        X_rearranged = np.zeros((X.shape[0], N_sub_patches, N_features))
        
        for i in range(X.shape[0]):
            for j in range(N_sub_patches):
                xj = X[i,j*L_sub_patch:(j+1)*L_sub_patch]
                X_rearranged[i,j,:] = [xj.mean(), xj.std(), np.double(scipy.stats.skew(xj)), 
                                       np.double(scipy.stats.kurtosis(xj)), xj.min(), xj.max(), 
                                       np.quantile(xj, 0.25), np.quantile(xj, 0.5), np.quantile(xj, 0.75), 
                                       np.quantile(xj, 0.75) - np.quantile(xj, 0.25)]
        return X_rearranged

    X_train_rearranged = rearranging_X(X_train); X_valid_rearranged = rearranging_X(X_valid)

    X_train = th.from_numpy(X_train_rearranged); X_valid = th.from_numpy(X_valid_rearranged)
    Y_train = th.from_numpy(Y_train); Y_valid = th.from_numpy(Y_valid)
    Y_train = th.unsqueeze(Y_train, -1); Y_valid = th.unsqueeze(Y_valid, -1)

    return X_train, Y_train, X_valid, Y_valid