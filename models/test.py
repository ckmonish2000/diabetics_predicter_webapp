from model import Model
import torch
import torch.nn as nn
from acc import accuracy
import pandas as pd


# data=pd.read_csv("diabetes2.csv")
# data=data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]]
# print(data.describe())


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



min=[0.000000 ,   0.000000,       0.000000,       0.000000 ,   0.000000,    0.000000,   21.000000]
max=[17.000000,  199.000000,     122.000000,      99.000000,  846.000000,   67.100000,   81.000000]


