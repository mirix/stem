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

np.random.seed(None)

df = pd.read_csv(cwd + '/Percentage_Female_Graduates_STEM_Descriptors.csv')
df_nn = df.dropna()
	
sujects = {
	'STEM': 'STEM',
	'SCI_MATH': 'Sciences and Maths',
	'ENG': 'Engineering',
	'ICT': 'ICT'	
}

descriptors = {
	'GDPPC': 'GDP per capita',
	'GINI': 'Gini Index',
	'HDI': 'Human Development Index',
	'IHDI': 'Indequality-adjusted Human Development Index',
	'GDI': 'Gender Development Index',
	'FHDI': 'Human Development Index (Women)',
	'GII': 'Gender Inequality Index',
	'stemt': 'Total % STEM graduates',
	'GSNI1': 'Gender Social Norms Index - 1 bias or more',
	'GSNI2': 'Gender Social Norms Index - 2 biasses or more',
	'GSNI0': 'Gender Social Norms Index - no bias',
	'GSNI_pol': 'Gender Social Norms Index - Political bias', 
	'GSNI_eco': 'Gender Social Norms Index - Economic bias',
	'GSNI_edu': 'Gender Social Norms Index - Educational bias',
	'GSNI_phi': 'Gender Social Norms Index - Physical integrity bias',
	'SCI_BENEFITS-Enthusiast': 'Science Benefits - Enthusiast',
	'SCI_BENEFITS-Excluded': 'Science Benefits - Excluded',
	'SCI_BENEFITS-Included': 'Science Benefits - Included',
	'SCI_BENEFITS-Sceptic': 'Science Benefits - Sceptic',
	'SCI_JOBS-Decrease': 'Science jobs expected to decrease',
	'SCI_JOBS-Increase': 'Science jobs expected to increase',
	'SCI_JOBS-Neither': 'Science jobs expected to remain invariable',
	'SCI_TRUST-High': 'Hight trust in science',
	'SCI_TRUST-Low': 'Low trust in science',
	'SCI_TRUST-Medium': 'Medium trust in science',
	'unemployment': 'Unemployment rate',
	'Christians': '% Christians',
	'Muslims': '% Muslims',
	'Unaffiliated': '% Non-religious'	
}

for sub in df.iloc[:, 2:6]:
	
	df_fi = pd.DataFrame.from_dict(descriptors, orient='index')
	df_fi.drop(df_fi.columns[0],axis=1,inplace=True)
	
	
	for n in range(1, 101):
		
		training, testing = train_test_split(df_nn, test_size=0.2)
		X_train = training[descriptors].values # Descriptors training
		X_test = testing[descriptors].values # Descriptors testing
		y_train = training[sub].values # Real % Female Graduates
		y_test = testing[sub].values # Real % Female Graduates
		
		### Feature Importance ###
		rgs.fit(X_train, y_train) 
		rgs.score(X_test, y_test)
		
		X_index = training[descriptors]
		fi = pd.DataFrame(rgs.feature_importances_, index = X_index.columns, columns=['imp'])
		df_fi = df_fi.merge(fi, how='outer', left_index=True, right_index=True)
		
	df_fi['importance'] = df_fi.mean(axis=1)
	df_fi = df_fi[['importance']].sort_values('importance', ascending=False)
	df_fi['features'] = df_fi.index
	df_fi.to_csv(cwd + '/Features_' + sub + '.csv', index=False)
	
	
