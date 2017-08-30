import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd

dataTrain = pd.read_csv("/Users/senora/Desktop/train.csv")
dataTest = pd.read_csv("/Users/senora/Desktop/test.csv")

x_train = dataTrain[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', 'AREA UNDER CULTIVATION']]
y_train = dataTrain[['YIELD']]

x_test = dataTest[['METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', 'AREA UNDER CULTIVATION']]
y_test = dataTest[['YIELD']]

ols = linear_model.LinearRegression()
model = ols.fit(x_train, y_train)
accuracy = ols.score(x_test, y_test)

print(accuracy)
print (model.predict(x_test)[0:5])