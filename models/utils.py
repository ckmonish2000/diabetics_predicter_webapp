import torch 
import torch.nn as nn
from models.acc import accuracy
from models.model import Model
import numpy as np
model=Model()


def predict(input):
    input=np.array(input)
    input=torch.from_numpy(input).float()
    pred=model(input)
    pred=torch.round(pred)
    if(pred>=0.5):
        pred=True
    else:
        pred=False
    return pred
