### Check accuracy during training

import torch as th
import torch.nn as nn

def check_accuracy(X_valid, Y_valid, model):
    
    device = 'cuda'
    N_valid = X_valid.shape[0]
    real_ttf = th.squeeze(Y_valid)
    loss_fn = nn.MSELoss()
    
    with th.no_grad():

        scores = th.squeeze(model(X_valid.to(device))).cpu()
        valid_loss_ = loss_fn(scores, real_ttf).cpu()
        valid_differences = scores[:] - real_ttf[:]
        valid_differences = valid_differences.cpu().numpy()
        valid_loss = valid_loss_.item()
    
    return valid_loss, valid_differences