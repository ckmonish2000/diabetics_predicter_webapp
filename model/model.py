import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset



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
y_train=torch.from_numpy(y_train)
X_test=torch.from_numpy(X_test).float()
y_test=torch.from_numpy(y_test)

class Model(nn.Module):
    """Some Information about Model"""
    def __init__(self):
        super().__init__()
        self.linear=nn.Sequential(
            nn.Linear(7,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.ReLU(),
            nn.Linear(2,1)            
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x=self.linear(x)
        x=self.sigmoid(x)
        return x


model=Model()
tl=TensorDataset(X_train,y_train)
dl=DataLoader(tl,batch_size=128,shuffle=True)

# testing dataloader and model
for i,j in dl:
    # print(i)
    # print(j)
    pred=model(i)
    print(pred)
    break

