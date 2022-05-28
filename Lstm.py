# LSTM Architecture

import torch as th
import torch.nn as nn

class Lstm(nn.Module):
    def __init__(self, N_features, hidden_size, num_layers, seq_length):
        super(Lstm, self).__init__()

        self.num_layers = num_layers #number of layers
        self.N_features = N_features # Number of features
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(N_features=N_features, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm => Input Shape : (N_batch, L_seq, N_feature)
    
    def forward(self,x):
        
        h_0 = th.zeros(self.num_layers, x.size(0), self.hidden_size) #hidden state
        c_0 = th.zeros(self.num_layers, x.size(0), self.hidden_size) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x.float(), (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = hn
        return out