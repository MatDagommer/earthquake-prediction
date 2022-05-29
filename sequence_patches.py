### Function to create patched sequences with some overlap out of the training data

import numpy as np
import torch as th

# Overlap is expressed as a fraction of patch size
def sequence_patches(size_patch, X, Y, overlap_rate):
    
    overlap = int(size_patch*overlap_rate)
    L = X.shape[0] # total length of the training acoustic signal
    n_patch = int(np.floor((L-size_patch)/(size_patch-overlap)+1))
    ids_no_seism = []

    X_patch = th.zeros(n_patch,size_patch)
    Y_patch = th.zeros(n_patch)
    
    for i in range (n_patch):
        X_patch[i,:] = X[i*(size_patch-overlap):(i+1)*size_patch-i*overlap] 
        Y_patch[i] = Y[(i+1)*size_patch - i*overlap] 
    
        # Remove patches with no seism
        if th.min(Y[i*(size_patch-overlap):(i+1)*size_patch - i*overlap]) > 0.001:
            ids_no_seism.append(i)
    
    X_patch = X_patch[ids_no_seism]
    Y_patch = Y_patch[ids_no_seism]
    
    return(n_patch, X_patch, Y_patch)