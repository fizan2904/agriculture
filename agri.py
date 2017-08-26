import numpy as np
import pandas, csv
import xlrd, math, datetime, pickle
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

years = []
MDs = []
HDs = []
ADs = []
Areas = []
cryield = []
with open('data.csv') as data:
	readCSV = csv.reader(data, delimiter=',')
	for row in readCSV:
		year = row[0]
		md = row[1]
		hd = row[2]
		ad = row[3]
		area = row[4]
		cr = row[5]
		years.append(year)
		MDs.append(md)
		HDs.append(hd)
		ADs.append(ad)
		Areas.append(area)
		cryield.append(cr)
years.pop(0)
MDs.pop(0)
HDs.pop(0)
ADs.pop(0)
Areas.pop(0)
cryield.pop(0)

years = list(map(float, years))
MDs = list(map(float, MDs))
HDs = list(map(float, HDs))
ADs = list(map(float, ADs))
Areas = list(map(float, Areas))
cryield = list(map(float, cryield))
X = [cryield, MDs, HDs, ADs, Areas]
forcast_col = 'cryield'
forcast_out = int(math.ceil(0.01*len(X)))
X['label'] = X[forcast_col].shift(-forcast_out)
print(X)
avg = []
hl = []
for i in range(0, len(years)):
	average = (MDs[i] + HDs[i] + ADs[i])/3.0
	avg.append(average)

for i in range(0, len(avg)):
	hilo = (avg[i] - Areas[i])/Areas[i] * 100.0
	hl.append(hilo)

#avg = np.array(avg)
#hl = np.array(hl)

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, cryield, test_size = 0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
forecase_set = clf.predict(X_lately)

#print(forecase_set, accuracy, forcast_out)

print(len(X_train))
print(len(Y_train))