#Inspired from https://cnvrg.io/pytorch-lstm/

import torch as th
import torch.nn as nn

class Lstm(nn.Module):
    
    def __init__(self, N_features, hidden_size, num_layers, seq_length):
        super(Lstm, self).__init__()

        self.num_layers = num_layers # number of layers
        self.N_features = N_features # number of features
        self.hidden_size = hidden_size # hidden state
        self.seq_length = seq_length # sequence length

        self.lstm = nn.LSTM(input_size=N_features, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm => Input Shape : (N_batch, L_seq, N_feature)
    
    def forward(self,x):
        
        output, (_,_) = self.lstm(x.float())
        out = output[:,-1,0] # Retrieving predicted time at the end of the training
        
        return out