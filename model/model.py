import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


data=pd.read_csv("diabetes2.csv")


# print(data.columns)

features=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]


y=data.pop("Outcome")
X=data[features]

# 
print("features: \n")
print(f"{X.head(5)}")
print("labels : \n")
print(f"{y.head(5)}")
