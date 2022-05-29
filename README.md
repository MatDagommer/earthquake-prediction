# Welcome to the eartquake prediction project !

Deep Learning Class Project at ESPCI under the supervision of Prof. Alexandre Allauzen.

## Description

Kaggle LANL Eartquake Prediction challenge : https://www.kaggle.com/c/LANL-Earthquake-Prediction

## Run the code

To download the repo via git, type:

    git clone https://github.com/MatDagommer/earthquake-prediction.git
  
Check that you are able to launch python from your Git Bash shell. 

    python --version

If that's not the case, this blog explains how to set it up : https://prishitakapoor2.medium.com/configuring-git-bash-to-run-python-for-windows-a624aa4ae2c5

## Train a new model

In the Git Bash, you can run the program with default parameters with the following command. 

You need to choose a model: 0 stands for 1D CNN and 1 for LSTM.

    python main.py -model 0
    
There are several parameters you can play with to customize the training. You can get the list of these arguments by typing:

    python main.py --help
    
For instance, if you want to train a 1D CNN with 1000 epochs, a batch size of 300 and a learning rate of 0.001, you can type:
    
    python main.py -ep 1000 -bs 300 -lr 0.0001
