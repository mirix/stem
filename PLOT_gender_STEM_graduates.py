#!/usr/bin/env python3

import os
cwd = os.getcwd()

import pandas as pd  
import numpy as np
from numpy.polynomial.polynomial import polyfit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(cwd + '/Percentage_Female_Graduates_STEM_Descriptors.csv', sep=',')

for col in  df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_nn = df.dropna()

print('Subject,Descriptor,Rsquared', file=open(cwd + '/images/RSquared.csv', 'w'))

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
	'IHDI': 'Inequality-adjusted Human Development Index',
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

plt.rcParams['figure.figsize'] = (12,6)

for sub in df_nn.iloc[:, 2:6]:
	
	for des in df_nn.iloc[:, 6:]: 
		
		b, m = polyfit(df_nn[des], df_nn[sub], 1)
		ax = df_nn.plot(x=des, y=sub, style='o')
		plt.plot(df_nn[des], b + m * df_nn[des], '-')
		df_nn[[des, sub, 'Country']].apply(lambda row: ax.text(*row),axis=1);
#		plt.title('% Female ' + sujects[sub] + ' Graduates vs ' + descriptors[des] + ' | ' + 'Rsq = %0.3f' % r2_score(df_nn[des], df_nn[sub]))
		plt.title('sciting.eu', color='#22697f')
		plt.xlabel(descriptors[des] + ' | ' + 'RSq = %0.3f' % r2_score(df_nn[des], df_nn[sub]))  
		plt.ylabel('% Female ' + sujects[sub] + ' Graduates')  
		
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.xaxis.grid(True)
		ax.yaxis.grid(True)
		ax.set_axisbelow(True)
		ax.legend().set_visible(False)
		
		plt.savefig(cwd + '/images/Female_' + sub + '_Graduates_vs_' + des + '.png', dpi=300, bbox_inches='tight')
		print(sub + ',' + des + ',' + str(r2_score(df_nn[des], df_nn[sub])), file=open(cwd + '/images/RSquared.csv', 'a'))
		plt.close()

plt.rcParams['figure.figsize'] = (14,8)
plt.rcParams['xtick.labelsize'] = 6

for sub in df.iloc[:, 2:6]:
	
	dfs = df.sort_values(sub, ascending=False)
	dfs.dropna(subset=[sub], inplace=True)
	fig, ax = plt.subplots()
	plt.bar(dfs['Country'], dfs[sub], data=None)
#	plt.title('% Female ' + sujects[sub] + ' Graduates per country') 
	plt.title('sciting.eu', color='#22697f') 
#	plt.xlabel(descriptors[des])
	plt.ylabel('% Female ' + sujects[sub] + ' Graduates')
#	plt.xlabel('sciting.eu', color='#22697f')
	
	plt.xticks(rotation=70, ha='right', rotation_mode='anchor')
	plt.tight_layout()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.grid(False)
	ax.yaxis.grid(True)
	ax.set_axisbelow(True)
#	plt.legend().set_visible(False)
	
	plt.savefig(cwd + '/images/Female_' + sub + '_Graduates_per_Country.png', dpi=300, bbox_inches='tight')
	plt.close()
