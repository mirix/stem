#!/usr/bin/env python3

import os
cwd = os.getcwd()

import pandas as pd  
import numpy as np
import sympy

df = pd.read_csv(cwd + '/Percentage_Female_Graduates_STEM_Descriptors.csv')

df_nn = df.dropna()
dfs = df_nn.iloc[:, 2:]

reduced_form, inds = sympy.Matrix(dfs).rref()
indx = np.array(inds)
indx = indx + 0
inds = indx.tolist()
dfsr = dfs.iloc[:, inds]

dfsr.to_csv(cwd + '/linearly_independent.csv', index=False)
