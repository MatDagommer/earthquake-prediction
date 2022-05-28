### Plotting Graphs

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew
import pickle

def plot_results(train_losses, valid_losses, best_mvd, best_mtd, min_tl, min_vl, mm, model_name):

    best_epoch = np.fromiter(valid_losses, dtype=float).argmin()
    
    fig, ax = plt.subplots(1, 2, figsize = (20,10))
    plt.rcParams['font.size'] = '20'
    ax[0].set(title = "Losses: Training and Validation") 
    ax[0].set_xlabel("epochs", fontsize = 20)
    ax[0].set_ylabel("MSE", fontsize = 20)
    ax[0].plot(train_losses,"r", label = "Training", linewidth = 3)
    ax[0].plot(valid_losses, "b", label = "Validation", linewidth = 3)
    ax[0].legend(loc = "upper right", fontsize = 18)
    ax[0].axvline(x=int(best_epoch), color = 'black', linestyle ="--", linewidth = 3)
    ax[0].annotate("Best epoch: {}\nMSE_train: {:.3f}\nMSE_valid: {:.3f}".format(int(best_epoch), min_tl, min_vl), \
                   xy = (0.5,0.5), xycoords = 'axes fraction')
    
    best_mvd_plot = mm.inverse_transform(best_mvd.reshape(-1, 1))
    best_mtd_plot = mm.inverse_transform(best_mtd.reshape(-1, 1))

    N_valid = best_mvd_plot.shape[0]
    
    mean = float(np.mean(best_mvd_plot))
    std_dev = float(np.std(best_mvd_plot))
    kurt = float(kurtosis(best_mvd_plot))
    skewn = float(skew(best_mvd_plot))
    q1 = float(np.quantile(best_mvd_plot, 0.25))
    median = float(np.quantile(best_mvd_plot, 0.5))
    q3 = float(np.quantile(best_mvd_plot, 0.75))
    mae = np.absolute(best_mvd_plot).sum() / N_valid
    mse = np.square(best_mvd_plot).sum() / N_valid
    
    text = "mean: {:.3f}\nstd: {:.3f}\nkurt: {:.3f}\nskew: {:.3f}\nq1: {:.3f}\nmed: {:.3f}\nq3: {:.3f}\niqr: {:.3f}\nmae: {:.3f}\nmse: {:.3f}\n".format(mean, std_dev, kurt, skewn, q1, median, q3, q3-q1, mae, mse)
    
    ax[1].hist(best_mvd_plot, alpha = 0.3, label = "validation set", bins = 100, density = True, range = (-16, 16))
    ax[1].set(title = "TTF error distributions at best epoch")
    ax[1].set_xlabel("Error (seconds)", fontsize = 20)
    ax[1].set_ylabel("Density", fontsize = 20)
    ax[1].hist(best_mtd_plot, alpha = 0.3, label = "training set", bins = 100, density = True, range = (-16, 16))
    ax[1].annotate(text, xy =(-15, 0.1))
    ax[1].legend()


    plt.gcf()
    plt.savefig('Models/' + model_name + '/' + model_name + "_plot.jpg")
    plt.show()

