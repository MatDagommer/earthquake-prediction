import numpy as np
import scipy

def feature_expansion(X, N_sub_patches, L_sub_patch):
    
    #print(type(X))
    N_samples = X.shape[0]
    x = X.reshape(N_samples, N_sub_patches, L_sub_patch)
    x_mean = np.mean(x, axis = 2).reshape(N_samples, N_sub_patches, 1)
    x_std = np.std(x, axis = 2).reshape(N_samples, N_sub_patches, 1)
    x_skew = np.array(scipy.stats.skew(x, axis = 2), dtype = np.double).reshape(N_samples, N_sub_patches, 1)
    x_kurt = np.array(scipy.stats.kurtosis(x, axis = 2), dtype = np.double).reshape(N_samples, N_sub_patches, 1)
    x_min = np.min(x, axis = 2).reshape(N_samples, N_sub_patches, 1)
    x_max = np.max(x, axis = 2).reshape(N_samples, N_sub_patches, 1)
    x_q1 = np.quantile(x, 0.25, axis = 2).reshape(N_samples, N_sub_patches, 1)
    x_med = np.quantile(x, 0.5, axis = 2).reshape(N_samples, N_sub_patches, 1)
    x_q3 = np.quantile(x, 0.75, axis = 2).reshape(N_samples, N_sub_patches, 1)
    x_iqr = x_q3 - x_q1
    
    X_rearranged = np.concatenate((x_mean, x_std, x_skew, x_kurt, x_min, x_max, x_q1, x_med, x_q3, x_iqr), axis = 2)
    return X_rearranged