import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

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
data = {
	'prediction': np.array(model.predict(x_test)[0:5]),
	'years': np.array([2006, 2007, 2008, 2009, 2010])
}
print(data['prediction'])
plt.plot(data['years'], data['prediction'])
plt.show()