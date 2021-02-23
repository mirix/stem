#!/usr/bin/env python3

import os
cwd = os.getcwd()

import pandas as pd
import numpy as np
import fuzzymatcher

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Percentage of female graduates in STEM (combined)

df_stem = pd.read_csv(cwd + '/5900119e-45a0-47ce-a4c0-15904d80ae6f_Data.csv', sep=',')

df_stem = df_stem.iloc[:, 3:].replace('"','').replace('..','NaN').replace('0','NaN').replace('100','NaN')
for col in  df_stem.columns[1:]:
    df_stem[col] = pd.to_numeric(df_stem[col], errors='coerce')
#df_stem.iloc[:, 1:] = pd.to_numeric(df_stem.iloc[:, 1:].stack(), errors='coerce').unstack()
df_stem.iloc[:, 1:] = df_stem.iloc[:, 1:].ffill(axis=1)
df_stem = df_stem.iloc[:, [0, -1]]
df_stem.dropna(how='all', inplace = True)
df_stem.columns = ['Code', 'STEM']
df_stem['Code'] = df_stem['Code'].astype(str)

df_stem['STEM'].replace('', np.nan, inplace=True)
df_stem.dropna(subset=['STEM'], inplace=True)
df_stem = df_stem.drop_duplicates()

# Percentage of female graduates in Natural Sciences and Maths (SCI_MAT)

df_sm = pd.read_csv(cwd + '/504e1314-57f0-4e51-a723-8245a0bda6c1_Data.csv', sep=',')

df_sm = df_sm.iloc[:, 3:].replace('"','').replace('..','NaN').replace('0','NaN').replace('100','NaN')
for col in  df_sm.columns[1:]:
    df_sm[col] = pd.to_numeric(df_sm[col], errors='coerce')
#df_sm.iloc[:, 1:] = pd.to_numeric(df_sm.iloc[:, 1:].stack(), errors='coerce').unstack()
df_sm.iloc[:, 1:] = df_sm.iloc[:, 1:].ffill(axis=1)
df_sm = df_sm.iloc[:, [0, -1]]
df_sm.dropna(how='all', inplace = True)
df_sm.columns = ['Code', 'SCI_MATH']
df_sm['Code'] = df_sm['Code'].astype(str)

df_sm['SCI_MATH'].replace('', np.nan, inplace=True)
df_sm.dropna(subset=['SCI_MATH'], inplace=True)
df_sm = df_sm.drop_duplicates()

# Percentage of female graduates in Engineering (ENG)

df_en = pd.read_csv(cwd + '/c7559abe-e1d9-4cfe-92a4-9c0fbac458ac_Data.csv', sep=',')

df_en = df_en.iloc[:, 3:].replace('"','').replace('..','NaN').replace('0','NaN').replace('100','NaN')
for col in  df_en.columns[1:]:
    df_en[col] = pd.to_numeric(df_en[col], errors='coerce')
#df_en.iloc[:, 1:] = pd.to_numeric(df_en.iloc[:, 1:].stack(), errors='coerce').unstack()
df_en.iloc[:, 1:] = df_en.iloc[:, 1:].ffill(axis=1)
df_en = df_en.iloc[:, [0, -1]]
df_en.dropna(how='all', inplace = True)
df_en.columns = ['Code', 'ENG']
df_en['Code'] = df_en['Code'].astype(str)

df_en['ENG'].replace('', np.nan, inplace=True)
df_en.dropna(subset=['ENG'], inplace=True)
df_en = df_en.drop_duplicates()

# Percentage of female graduates in IT (ICT)

df_it = pd.read_csv(cwd + '/f88a3b40-bd07-4b87-8a63-b80fa2704e27_Data.csv', sep=',')

df_it = df_it.iloc[:, 3:].replace('"','').replace('..','NaN').replace('0','NaN').replace('100','NaN')
for col in  df_it.columns[1:]:
    df_it[col] = pd.to_numeric(df_it[col], errors='coerce')
#df_it.iloc[:, 1:] = pd.to_numeric(df_it.iloc[:, 1:].stack(), errors='coerce').unstack()
df_it.iloc[:, 1:] = df_it.iloc[:, 1:].ffill(axis=1)
df_it = df_it.iloc[:, [0, -1]]
df_it.dropna(how='all', inplace = True)
df_it.columns = ['Code', 'ICT']
df_it['Code'] = df_it['Code'].astype(str)

df_it['ICT'].replace('', np.nan, inplace=True)
df_it.dropna(subset=['ICT'], inplace=True)
df_it = df_it.drop_duplicates()

# MERGE Female Graduate Percentages (FGP)

df_fgp = df_stem.merge(df_sm, on='Code', how='outer').merge(df_en, on='Code', how='outer').merge(df_it, on='Code', how='outer')

# Human Development Index (HDI) and Inequality-adjusted Human Development Index (IHDI)

df_hdi = pd.read_excel(cwd + '/2020_Statistical_Annex_Table_3_IA_HDI.xlsx', sheet_name='Table 3')

df_hdi = df_hdi.iloc[5:, [0, 1, 2, 4]]
df_hdi.columns = ['Rank', 'Country', 'HDI', 'IHDI']
df_hdi['Rank'].replace('', np.nan, inplace=True)
df_hdi.dropna(subset=['Rank'], inplace=True)
df_hdi = df_hdi.iloc[:, 1:].replace(',','').replace('"','').replace('..','NaN')

for col in df_hdi.columns[1:]:
    df_hdi[col] = pd.to_numeric(df_hdi[col], errors='coerce')
df_hdi['Country'] = df_hdi['Country'].astype(str)

#print(df_hdi.dtypes)

# Gender Inequality Index (GII)

df_gii = pd.read_excel(cwd + '/2020_Statistical_Annex_Table_5_GII.xlsx', sheet_name='Table 5')

df_gii = df_gii.iloc[7:, [0, 1, 2]]
df_gii.columns = ['Rank', 'Country', 'GII']
df_gii['Rank'].replace('', np.nan, inplace=True)
df_gii.dropna(subset=['Rank'], inplace=True)
df_gii = df_gii.iloc[:, 1:].replace(',','').replace('"','').replace('..','NaN')

for col in df_gii.columns[1:]:
    df_gii[col] = pd.to_numeric(df_gii[col], errors='coerce')
df_gii['Country'] = df_gii['Country'].astype(str)

# Gender Development Index (GDI) and Female Human Development Index (FHDI)

df_gdi = pd.read_excel(cwd + '/2020_Statistical_Annex_Table_4_GDI.xlsx', sheet_name='Table 4')

df_gdi = df_gdi.iloc[7:, [0, 1, 2, 6]]
df_gdi.columns = ['Rank', 'Country', 'GDI', 'FHDI']
df_gdi['Rank'].replace('', np.nan, inplace=True)
df_gdi.dropna(subset=['Rank'], inplace=True)
df_gdi = df_gdi.iloc[:, 1:].replace(',','').replace('"','').replace('..','NaN')

for col in df_gdi.columns[1:]:
    df_gdi[col] = pd.to_numeric(df_gdi[col], errors='coerce')
df_gdi['Country'] = df_gdi['Country'].astype(str)

# MERGE Various Indexes (vis)

df_vis = df_hdi.merge(df_gdi, on='Country', how='outer').merge(df_gii, on='Country', how='outer')

# GDP per capita (GDPPC)

df_gdp = pd.read_csv(cwd + '/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2017902.csv', sep=',', header=None, skiprows=5)

df_gdp = df_gdp.replace('"','')
for col in  df_gdp.columns[4:]:
    df_gdp[col] = pd.to_numeric(df_gdp[col], errors='coerce')
#df_stem.iloc[:, 1:] = pd.to_numeric(df_stem.iloc[:, 1:].stack(), errors='coerce').unstack()
df_gdp.iloc[:, 4:] = df_gdp.iloc[:, 4:].ffill(axis=1)
df_gdp = df_gdp.iloc[:, [0, 1, -1]]
df_gdp.columns = ['Country', 'Code', 'GDPPC']

# Gini Index (GINI)

df_gini = pd.read_csv(cwd + '/API_SI.POV.GINI_DS2_en_csv_v2_2015287.csv', sep=',', header=None, skiprows=5)

df_gini = df_gini.replace('"','')
for col in  df_gini.columns[4:]:
    df_gini[col] = pd.to_numeric(df_gini[col], errors='coerce')
#df_stem.iloc[:, 1:] = pd.to_numeric(df_stem.iloc[:, 1:].stack(), errors='coerce').unstack()
df_gini.iloc[:, 4:] = df_gini.iloc[:, 4:].ffill(axis=1)
df_gini = df_gini.iloc[:, [0, 1, -1]]
df_gini.columns = ['Country', 'Code', 'GINI']

# MERGE GDP and GINI (ginip)

df_ginip = df_gdp.merge(df_gini, on='Code', how='outer')

df_ginip = df_ginip[['Country_x', 'Code', 'GDPPC', 'GINI']]
df_ginip.columns = ['Country', 'Code', 'GDPPC', 'GINI']

# MERGE FGP and GINIP (fgin)

df_fgin = df_fgp.merge(df_ginip, on='Code', how='left')
df_fgin = df_fgin[['Country', 'Code', 'STEM', 'SCI_MATH', 'ENG', 'ICT', 'GDPPC', 'GINI']]

# MERGE fgin and vis

df_fgin['Country'] = df_fgin['Country'].str.replace('Czech Republic', 'Czechia')
df_fgin['Country'] = df_fgin['Country'].str.replace('Kyrgyz Republic', 'Kyrgyzstan')
df_fgin['Country'] = df_fgin['Country'].str.replace('Slovak Republic', 'Slovakia')
df_fgin['Country'] = df_fgin['Country'].str.replace('Congo, Dem. Rep.', 'Congo Democratic Republic', regex=False)
df_fgin['Country'] = df_fgin['Country'].str.replace('Congo, Rep.', 'Congo', regex=False)
df_fgin['Country'] = df_fgin['Country'].str.replace('Egypt, Arab Rep.', 'Egypt', regex=False)
df_fgin['Country'] = df_fgin['Country'].str.replace('Iran, Islamic Rep.', 'Iran', regex=False)
df_fgin['Country'] = df_fgin['Country'].str.replace('Korea, Rep.', 'Korea', regex=False)
df_fgin['Country'] = df_fgin['Country'].str.replace('Gambia, The', 'Gambia')
df_fgin['Country'] = df_fgin['Country'].str.replace('Syrian Arab Republic', 'Syria')

df_vis['Country'] = df_vis['Country'].str.replace('Viet Nam', 'Vietnam')
df_vis['Country'] = df_vis['Country'].str.replace('Iran (Islamic Republic of)', 'Iran', regex=False)
df_vis['Country'] = df_vis['Country'].str.replace('Korea (Republic of)', 'Korea', regex=False)
df_vis['Country'] = df_vis['Country'].str.replace('Moldova (Republic of)', 'Moldova', regex=False)
df_vis['Country'] = df_vis['Country'].str.replace('Eswatini (Kingdom of)', 'Eswatini', regex=False)
df_vis['Country'] = df_vis['Country'].str.replace('Congo (Democratic Republic of the)', 'Congo Democratic Republic', regex=False)
df_vis['Country'] = df_vis['Country'].str.replace('Lao People\'s Democratic Republic', 'Lao')
df_vis['Country'] = df_vis['Country'].str.replace('Syrian Arab Republic', 'Syria')

protectorates = ['PRK','MAC','PRI','SMR','SXM','PSE','ABW','BMU','AND','ATG','LIE','ARG']
df_fgin = df_fgin[~df_fgin.Code.isin(protectorates)]

df = fuzzymatcher.fuzzy_left_join(df_fgin, df_vis, left_on = 'Country', right_on = 'Country')

df = df[['Country_right','Code','STEM','SCI_MATH','ENG','ICT','GDPPC','GINI','HDI','IHDI','GDI','FHDI','GII']]
df.columns = ['Country','Code','STEM','SCI_MATH','ENG','ICT','GDPPC','GINI','HDI','IHDI','GDI','FHDI','GII']

# Percentage of STEM graduates (stemt)

df_stemt = pd.read_csv(cwd + '/NATMON_DS_18022021070738722.csv', sep=',', usecols=['NATMON_IND', 'LOCATION', 'Value']).replace('"','')

filter = df_stemt['NATMON_IND'] == 'FOSGP_5T8_F500600700'
df_stemt = df_stemt[filter]
df_stemt = df_stemt.groupby(['LOCATION']).tail(1)

df_stemt = df_stemt[['LOCATION', 'Value']]
df_stemt.columns = ['Code', 'stemt']

df_stemt['stemt'] = pd.to_numeric(df_stemt['stemt'], errors='coerce')

# MERGE stemt

df = df.merge(df_stemt, on='Code', how='left')

# Gender Social Norms Index (GSNI)

df_gsni = pd.read_excel(cwd + '/gsni_tables.xlsx', sheet_name='TA1', header=None, skiprows=8)
df_gsni.drop(df_gsni.tail(13).index, inplace=True)

for col in  df_gsni.columns[1:]:
    df_gsni[col] = pd.to_numeric(df_gsni[col], errors='coerce')

df_gsni.columns = ['Country', 'Period', 'GSNI1', 'GSNI2', 'GSNI0', 'GSNI_pol', 'GSNI_eco', 'GSNI_edu', 'GSNI_phi']
df_gsni = df_gsni[['Country', 'GSNI1', 'GSNI2', 'GSNI0', 'GSNI_pol', 'GSNI_eco', 'GSNI_edu', 'GSNI_phi']]

# MERGE GSNI

df_gsni['Country'] = df_gsni['Country'].str.replace('Korea (Republic of)', 'Korea', regex=False)
df_gsni['Country'] = df_gsni['Country'].str.replace('Moldova, Republic of', 'Moldova')
df_gsni['Country'] = df_gsni['Country'].str.replace('Iran, Islamic Republic of', 'Iran')
df_gsni['Country'] = df_gsni['Country'].str.replace('Palestine, State of', 'Palestine')
df_gsni['Country'] = df_gsni['Country'].str.replace('Trinidad and Tobago', 'Trinidad Tobago')
df_gsni['Country'] = df_gsni['Country'].str.replace('Viet Nam', 'Vietnam')

df['Country'] = df['Country'].str.replace('United Arab Emirates', 'UAE')

df = fuzzymatcher.fuzzy_left_join(df, df_gsni, left_on = 'Country', right_on = 'Country')

df = df[['Country_left','Code','STEM','SCI_MATH','ENG','ICT','GDPPC','GINI','HDI','IHDI','GDI','FHDI','GII','stemt','GSNI1','GSNI2','GSNI0','GSNI_pol','GSNI_eco','GSNI_edu','GSNI_phi']]
df.columns = ['Country','Code','STEM','SCI_MATH','ENG','ICT','GDPPC','GINI','HDI','IHDI','GDI','FHDI','GII','stemt','GSNI1','GSNI2','GSNI0','GSNI_pol','GSNI_eco','GSNI_edu','GSNI_phi']

# Wellcome Trust (wct)

df_wct = pd.read_excel(cwd + '/wgm2018-dataset-crosstabs-all-countries.xlsx', sheet_name='Crosstabs all countries', skiprows=2, usecols=['Country','Question','Response','Column N %'])

df_wct['Question'].ffill(inplace = True)

jobs = 'Q19 Overall, do you think that science and technology will increase or decrease the number of jobs in your local area in the next five years?'
trust = 'Wellcome Global Monitor Trust in Scientists Index (recoded into 3 categories)'
benef = 'How a person views personal & societal benefit of science'

df_wct['Question'] = df_wct['Question'].replace({jobs:'SCI_JOBS', trust:'SCI_TRUST', benef:'SCI_BENEFITS'})

filter = df_wct.Question.str.contains('SCI_JOBS') | \
         df_wct.Question.str.contains('SCI_TRUST') | \
         df_wct.Question.str.contains('SCI_BENEFITS')
                
df_wct = df_wct[filter]

df_wct.rename(columns={'Column N %': 'Value'}, inplace=True)

todel = ['DK', 'Total', 'Did not answer', 'Refused']
df_wct = df_wct[~df_wct['Response'].str.contains('|'.join(todel))]

df_wct['Response'] = df_wct['Response'].str.replace('(Neither/Have no effect)', 'Neither', regex=False)

df_wct['WT'] = df_wct['Question'] + '-' + df_wct['Response']

df_wct = df_wct[['Country', 'WT', 'Value']]

df_wct['WT'] = df_wct['WT'].str.replace(' trust', '')

df_wct['Value'] = df_wct['Value'] * 100

df_wct = df_wct.pivot(index='Country', columns='WT', values='Value')
df_wct['Country'] = df_wct.index
df_wct = df_wct[['Country','SCI_BENEFITS-Enthusiast','SCI_BENEFITS-Excluded','SCI_BENEFITS-Included','SCI_BENEFITS-Sceptic','SCI_JOBS-Decrease','SCI_JOBS-Increase','SCI_JOBS-Neither','SCI_TRUST-High','SCI_TRUST-Low','SCI_TRUST-Medium']]

# MERGE WCT

df_wct['Country'] = df_wct['Country'].str.replace('Congo, Rep.', 'Congo', regex=False)
df_wct['Country'] = df_wct['Country'].str.replace('Czech Republic', 'Czechia')
df_wct['Country'] = df_wct['Country'].str.replace('United Arab Emirates', 'UAE')

df = fuzzymatcher.fuzzy_left_join(df, df_wct, left_on = 'Country', right_on = 'Country')
df.drop(['best_match_score', '__id_left', '__id_right', 'Country_right'], axis=1, inplace=True)
df = df.rename(columns={'Country_left': 'Country'})

# Unemployment rate (unem)

df_unem = pd.read_csv(cwd + '/API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_2055580.csv', skiprows=4)
df_unem = df_unem.replace('"','')

df_unem = df_unem[['Country Code', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']]
df_unem.columns = ['Code', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
for col in df_unem.columns[1:]:
    df_unem[col] = pd.to_numeric(df_unem[col], errors='coerce')
df_unem.dropna(how='any', inplace = True)

df_unem['unemployment'] = df_unem.iloc[:, 1:].mean(axis=1)
df_unem = df_unem[['Code', 'unemployment']]

# MERGE unem

df = df.merge(df_unem, on='Code', how='left')

# Religions (rel)

df_rel = pd.read_excel(cwd + '/Religious_Composition_by_Country_2010-2050.xlsx', sheet_name='rounded_percentage', usecols=['Country','Christians','Muslims','Unaffiliated'])

df_rel = df_rel.iloc[7:]
df_rel = df_rel.head(234)
df_rel = df_rel.replace('< 1.0', '0.0', regex=False).replace('>99.0', '100.0', regex=False)
for col in  df_rel.columns[1:]:
    df_rel[col] = pd.to_numeric(df_rel[col], errors='coerce')
    
# Merge rel

df_rel['Country'] = df_rel['Country'].str.replace('Democratic Republic of the Congo', 'Congo Democratic Republic')
df_rel['Country'] = df_rel['Country'].str.replace('Republic of the Congo', 'Congo')
df_rel['Country'] = df_rel['Country'].str.replace('Czech Republic', 'Czechia')
df_rel['Country'] = df_rel['Country'].str.replace('United Arab Emirates', 'UAE')
df_rel['Country'] = df_rel['Country'].str.replace('Laos', 'Lao')
df_rel['Country'] = df_rel['Country'].str.replace('Trinidad and Tobago', 'Trinidad Tobago')
df_rel['Country'] = df_rel['Country'].str.replace('Swaziland', 'Eswatini')
df_rel['Country'] = df_rel['Country'].str.replace('South Korea', 'Korea')
df_rel['Country'] = df_rel['Country'].str.replace('North Korea', 'Korea DR')
df_rel['Country'] = df_rel['Country'].str.replace('Republic of Macedonia', 'North Macedonia')

df = fuzzymatcher.fuzzy_left_join(df, df_rel, left_on = 'Country', right_on = 'Country')
df.drop(['best_match_score', '__id_left', '__id_right', 'Country_right'], axis=1, inplace=True)
df = df.rename(columns={'Country_left': 'Country'})

# Write dataset

df.to_csv(cwd + '/Percentage_Female_Graduates_STEM_Descriptors.csv', index=False)
