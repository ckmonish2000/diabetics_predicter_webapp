from model import Model
import torch
import torch.nn as nn
from acc import accuracy



model=Model()
loss=nn.BCELoss()
optim=torch.optim.SGD(model.parameters(),lr=0.0001)


def modeltest(dl):
    for i,j in dl:
        # print(i)
        # print(j)
        pred=model(i)
        # print(pred)
        print(f"acc={accuracy(pred,j)}")
        print(f"loss={loss(pred,j).item()}")
        break
