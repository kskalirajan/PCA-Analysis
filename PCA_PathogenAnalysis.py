# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:47:46 2021

@author: Kalirajan Sakthivel
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

## import data
my_csv = "pca.csv" ## path to your dataset

dat = pd.read_csv(my_csv, index_col=0, na_values=None)
# if no row or column titles in your csv, pass 'header=None' into read_csv
# and delete 'index_col=0' -- but your biplot will be clearer with row/col names


# Data information : Index, columns and its contents.....
print(dat.head())
print(dat.tail())
print("\nData Information : ")
dat.info()

print("\nNumber of Columns : ", len(dat.columns))
print("Number of rows : ", len(dat.index))


############# Correlation matrix - START ############
correlation_mat = dat.corr()
#print (correlation_mat)
sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of strains")

plt.xlabel("strains")
plt.ylabel("strains features")

plt.show()   
############# Correlation matrix - END ############

############# PCA Biplot - visualize projections - Library  ALGORITHM - Start ############ 

print("\n\nPCA BiPlot - Process .....")

# Import libraries
from pca import pca

#scaler = StandardScaler()
#scaler.fit(dat)
#dat=scaler.transform(dat)    

# Initialize
model = pca(n_components=0.95, n_feat=len(dat.index), alpha=0.05, n_std=2, onehot=False, normalize=False, detect_outliers=['ht2','spe'], random_state=None)

# Fit transform
out = model.fit_transform(dat)


# Print the top features. The results show that f1 is best, followed by f2 etc
print("\ntopfeat: \n",  out['topfeat'])
print("\noutliers:\n",  out['outliers'])
print("\nshape : \n",   dat.shape)


fig, ax = model.plot()
fig, ax = model.biplot(cmap=None, label=False, legend=False)

# Create only the scatter plots
print("\n\n\nCreate only the scatter plots")
fig, ax = model.scatter(legend=False, cmap='Set2')
fig, ax = model.scatter3d(legend=False, cmap='Set2')


# Create only the biPlots
print("\n\n\nCreate only the bi plots")
fig, ax = model.biplot(n_feat=len(dat.index), figsize=(10,8), legend=False, cmap='Set2')
fig, ax = model.biplot3d(n_feat=len(dat.index), figsize=(10,8), legend=True,cmap='Set2')

############# PCA Biplot - visualize projections - Library  ALGORITHM - End ############ 

