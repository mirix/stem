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

rgs = RandomForestRegressor(n_estimators=100, n_jobs=-1)

np.random.seed(None)

df = pd.read_csv(cwd + '/Percentage_Female_Graduates_STEM_Descriptors.csv')
df_nn = df.dropna()

subjects = ['STEM', 'SCI_MATH', 'ENG', 'ICT']

for sub in subjects:
	
	df_sub = pd.read_csv(cwd + '/Features_' + sub + '.csv')
	
	descriptors = df_sub['features'].tolist()
		
	print('Descriptor,RSquared', file=open(cwd + '/Descriptors_' + sub + '.txt', 'w'))
	
	rsqn = -1000000
	dmo = []
	
	for des in descriptors:
		
		rsql = []
		dmo.append(des)
		
		for n in range(1, 101):
			
			training, testing = train_test_split(df_nn, test_size=0.2)
			
			X_train = training[dmo].values # Descriptors
			X_test = testing[dmo].values # Descriptors
			y_train = training[sub].values # Real % Female ICT Graduates
			y_test = testing[sub].values # Real % Female ICT Graduates
			
			y_pred = rgs.fit(X_train, y_train).predict(X_test)
			
			rsq = r2_score(y_test, y_pred)
			rsql.append(rsq)
			
		rsqnn = np.mean(rsql)
		
		if rsqnn >= rsqn:
			rsqn = rsqnn
			print(des + ',' + str(rsqn), file=open(cwd + '/Descriptors_' + sub + '.txt', 'a'))
		else: 
			dmo = [x for x in descriptors if x != des]
	
	
