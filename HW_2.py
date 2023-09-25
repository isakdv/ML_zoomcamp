# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:43:40 2023

@author: Daria Isaak
"""


###########################################
import pandas as pd
import numpy as np
import wget
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
filename = wget.download(url)
df = pd.read_csv(filename)
print(df)

#Look at the median_house_value variable. Does it have a long tail
import seaborn as sns
sns.histplot(df['median_house_value']) #positibe tail on the right


#First, keep only the records where ocean_proximity is either '<1H OCEAN' or 'INLAND'

#updated_df = df[(df['ocean_proximity'] == '<1H OCEAN') | (df['ocean_proximity'] == 'INLAND')]
columns = ['latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income',
'median_house_value']

updated_df = df[(df['ocean_proximity'] == '<1H OCEAN') | (df['ocean_proximity'] == 'INLAND')][columns].reset_index()







###########################################
#Question 1
#There's one feature with missing values. What is it

updated_df.isnull().sum() #total_bedrooms


###########################################
#Question 2
#What's the median (50% percentile) for variable 'population'

updated_df['population'].quantile(0.50) #1195.0

# Prepare and split the dataset
# Shuffle the dataset (the filtered one you created above), use seed 42.
updated_df = updated_df.sample(frac=1, random_state=42).reset_index(drop=True)


# Split your data in train/val/test sets, with 60%/20%/20% distribution.
n = len(updated_df)

n_train = int(n*0.6)
n_val = int(n*0.2)
n_test = n - n_train - n_val

train = updated_df.iloc[:n_train]
val = updated_df.iloc[n_train:n_train + n_val]
test = updated_df.iloc[n_train + n_val:]


# Apply the log transformation to the median_house_value variable using the np.log1p() function
y_train = np.log1p(train.median_house_value.values)
y_val = np.log1p(val.median_house_value.values)
y_test = np.log1p(test.median_house_value.values)

###########################################
#Question 3

# We need to deal with missing values for the column from Q1.
# We have two options: fill it with 0 or with the mean of this variable.
# Try both options. For each, train a linear regression model without regularization using the code from the lessons.
# For computing the mean, use the training only!

#total_bedrooms
train_1 = train.copy()
train_2 = train.copy()

train_1['total_bedrooms'] = train['total_bedrooms'].fillna(0)
train_2['total_bedrooms'] = train['total_bedrooms'].fillna(train['total_bedrooms'].mean())

base = ['latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income']

X_train_1 = train_1[base].values
X_train_2 = train_2[base].values


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


# Use the validation dataset to evaluate the models and compare the RMSE of each option.
# Round the RMSE scores to 2 decimal digits using round(score, 2)
# Which option gives better RMSE?

val_1 = val.copy()
val_2 = val.copy()

val_1['total_bedrooms'] = val['total_bedrooms'].fillna(0)
val_2['total_bedrooms'] = val['total_bedrooms'].fillna(train['total_bedrooms'].mean())


X_val_1 = val_1[base].values
X_val_2 = val_2[base].values

y_pred_1 = train_linear_regression(X_train_1, y_train)[0] + X_val_1.dot(train_linear_regression(X_train_1, y_train)[1])
y_pred_2 = train_linear_regression(X_train_2, y_train)[0] + X_val_2.dot(train_linear_regression(X_train_2, y_train)[1])


def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)
    
score_1 = rmse(y_val, y_pred_1)
score_2 = rmse(y_val, y_pred_2)


print(round(score_1, 2)) #same
print(round(score_2, 2))

################################
#Question 4


def train_linear_regression_reg(X, y, r):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


r_list = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
rmse_list = []

for r in r_list:
    w0, w = train_linear_regression_reg(X_train_1, y_train, r)
    y_pred_z = w0 + X_val_1.dot(w)
    score_z = rmse(y_val, y_pred_z)
    rmse_list.append(score_z)


min_rmse = min(rmse_list)
index_of_min = rmse_list.index(min_rmse)
print((r_list[index_of_min], min_rmse))  #(0, 0.3409235996318668)



   













































































