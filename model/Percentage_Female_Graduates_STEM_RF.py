#!/usr/bin/env python3

import os
cwd = os.getcwd()

import pandas as pd  
import numpy as np

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
#rgs = RandomForestRegressor(n_estimators=1422, max_features=10, max_depth=20, minamplesplit=2, minamples_leaf=1, n_jobs=-1)
rgs = RandomForestRegressor(n_estimators=100, n_jobs=-1)

np.random.seed(0)

df = pd.read_csv(cwd + '/Percentage_Female_Graduates_STEM_Descriptors.csv')
df_nn = df.dropna()

training, testing = train_test_split(df_nn, test_size=0.2)
	
sujects = {
	'STEM': 'STEM',
	'SCI_MATH': 'Sciences and Maths',
	'ENG': 'Engineering',
	'ICT': 'ICT'	
}

descriptors = {
	'stemt': 'Total % STEM graduates',
	'SCI_JOBS-Decrease': 'Science jobs expected to decrease',
	'Unaffiliated': '% Non-religious',
	'SCI_TRUST-Low': 'Low trust in science'
}

X_train = training[descriptors].values # Descriptors training
X_test = testing[descriptors].values # Descriptors testing

print('Subject,RSquared', file=open(cwd + '/Percentage_Female_Graduates_RF.txt', 'w'))

for sub in df.iloc[:, 2:6]:
	
	y_train = training[sub].values # Real % Female Graduates
	y_test = testing[sub].values # Real % Female Graduates
	
	y_pred = rgs.fit(X_train, y_train).predict(X_test)

	print(sub + ',' + str(r2_score(y_test, y_pred)), file=open(cwd + '/Percentage_Female_Graduates_RF.txt', 'a'))
	
	b, m = polyfit(y_test, y_pred, 1)
	
	plt.plot(y_test, y_pred, '.')
	plt.plot(y_test, b + m * y_test, '-')
	plt.xlabel('Real % Female ' + sub + ' Graduates')
	plt.ylabel('Predicted % Female ' + sub +' Graduates')
	plt.title('RSq = %0.3f' % r2_score(y_test, y_pred))
	plt.savefig(cwd + '/Percentage_Female_' + sub + '_Graduates_RF.png', dpi=300, bbox_inches='tight')
	#plt.show()
	plt.close()
	
	
	### Feature Importance ###
	rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1) 
	rfr.fit(X_train, y_train) 
	rfr.score(X_test, y_test)
	
	X_index = training[descriptors]
	feature_importances = pd.DataFrame(rfr.feature_importances_, index = X_index.columns, columns=['importance']).sort_values('importance', ascending=False)
	print("Feature_Importance:\n", feature_importances, file=open(cwd + '/Percentage_Female_Graduates_' + sub + '_FI.txt', 'w'))
	
