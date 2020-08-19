import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from model import Model
from acc import  accuracy
from test import modeltest

data=pd.read_csv("diabetes2.csv")


# print(data.columns)

features=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]
print(len(features))


y=data.pop("Outcome")
X=data[features]

# display of data features and labels

# print("features: \n")
# print(f"{X.head(5)}")
# print("labels : \n")
# print(f"{y.head(5)}")


X=np.array(X)
y=np.array(y)
# train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

# convert to numpy tensors
X_train=torch.from_numpy(X_train).float()
y_train=torch.from_numpy(y_train).float().unsqueeze(dim=1)
X_test=torch.from_numpy(X_test).float()
y_test=torch.from_numpy(y_test).float().unsqueeze(dim=1)

# Dataloader
tl=TensorDataset(X_train,y_train)
dl=DataLoader(tl,batch_size=128,shuffle=True)
# val loader
tl=TensorDataset(X_test,y_test)
vl=DataLoader(tl,batch_size=128,shuffle=True)

# training reqs
model=Model()
loss=nn.BCELoss()



# testing dataloader and model

# modeltest(dl)

def fit(epochs,lr,model,loss,dl,vl):
    optim=torch.optim.SGD(model.parameters(),lr=lr, momentum=0.12)
    history=[]
    for epoch in range(epochs):
        for i,j in dl:
            pred=model(i)
            ls=loss(pred,j)
            ls.backward()
            optim.step()
        for i,j in vl:
            pred=model(i)
            ls=loss(pred,j)
            acc=accuracy(pred,j)
            print({'epoch.no':epoch,'loss':ls.item(),'acc':acc.item()})
            history.append({'epoch.no':epoch,'loss':ls.item(),'acc':acc.item()})
    return history

train=fit(700,1e-9,model,loss,dl,vl)
acc=[i["acc"] for i in train]
loss=[i["loss"] for i in train]

import statistics
print(f"acc_mean={statistics.mean(acc)}")
print(f"loss_mean={statistics.mean(loss)}")

torch.save(model.state_dict(),"./model.pth")










