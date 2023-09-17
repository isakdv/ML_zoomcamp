# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:44:53 2023

@author: Daria Isaak
"""
import pandas as pd
import numpy as np
pd.__version__

###########################################
import wget
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
filename = wget.download(url)
df = pd.read_csv(filename)

###########################################
df.shape[1] 
###########################################
df.columns[df.isna().any()]

##########################################
df['ocean_proximity'].drop_duplicates().tolist()

##########################################

reduced_df = df[df['ocean_proximity'] == 'NEAR BAY']
reduced_df['median_house_value'].mean()

#########################################
cc = df['total_bedrooms'].mean()
df['total_bedrooms'] = df['total_bedrooms'].fillna(cc)
ccc = df['total_bedrooms'].mean()
print(cc)
print(ccc)

##########################################
filtered_df = df[df['ocean_proximity'] == 'ISLAND'][['housing_median_age', 'total_rooms', 'total_bedrooms']]

x = filtered_df.values
xtx = x.T @ x
xtx_inv = np.linalg.inv(xtx)

y = np.array([950, 1300, 800, 1000, 1300])
w = xtx_inv @ x.T @y
print(w[-1])

