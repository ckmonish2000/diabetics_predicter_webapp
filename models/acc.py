import torch

def accuracy(pred,label):
    pred=torch.round(pred)
    return torch.tensor(torch.sum(pred==label).item()/torch.numel((label)))