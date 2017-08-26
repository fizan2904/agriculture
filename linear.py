import numpy as np
import pandas, csv
import xlrd, math, datetime, pickle
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

df = pandas.read_csv('data.csv',parse_dates=True)
df = df[['YIELD', 'METEOROLOGICAL DROUGHT', 'HYDROLOGICAL DROUGHT', 'AGRICULTURAL DROUGHT', 'AREA UNDER CULTIVATION']]
forcast_col = 'YIELD'
df.fillna(-99999, inplace=True)

forcast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forcast_col].shift(-forcast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forcast_out:]
X = X[:-forcast_out:]

df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.2)
clf = LinearRegression(n_jobs=-1)
#clf = svm.SVR(kernel = 'poly')
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
forecase_set = clf.predict(X_lately)

print(forecase_set, accuracy, forcast_out)

#df['Forecast'] = np.nan
#
#last_date = df.iloc[-1].name
#print(last_date)
#last_unix = last_date.timestamp()
#one_day = 86400
#next_unix = last_unix + one_day
#
#for i in forecase_set:
	#next_date = datetime.datetime.fromtimestamp(next_unix)
	#next_unix += one_day
	#df.loc[next_date] = [np.nan for _ in range(len(da.columns)-1)] + [i]
#
#df['YIELD'].plot()
#df['Forecast'].plot()
#plt.xlabel('Date')
#plt.ylabel('Yield')
#plt.show()