# 1 Dimension Convolutional Neural Network

import torch.nn as nn
import torch as th


class Cnn1d(nn.Module):
    def __init__(self,num_classes=1):
        super(Cnn1d, self).__init__()
        self.seq_1=nn.Sequential(nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 10, stride = 5),
                                 nn.ReLU(),
                                 nn.MaxPool1d(kernel_size = 2),
                                 nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 10, stride = 5, \
                                           padding = 'valid'),
                                 nn.ReLU(),
                                 nn.MaxPool1d(kernel_size = 2),
                                 nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 10, stride = 5, \
                                           padding = 'valid'),
                                 nn.ReLU()
                                )
        
        self.seq_2=nn.Sequential(nn.Dropout(p = 0.4),
                                 nn.Linear(in_features = 64, out_features = 1)
                                )
    def forward(self,x):
        x=self.seq_1(x)
        x = th.mean(x, dim = 2, keepdim = True)
        x = th.squeeze(x)
        x=self.seq_2(x)
        return (x)