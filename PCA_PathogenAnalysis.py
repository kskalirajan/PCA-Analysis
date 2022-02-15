# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:47:46 2021

@author: SKA4COB
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
#print(dat.head())
#print(dat.tail())
#print("\nData Information : ")
#dat.info()

print("\nNumber of Columns : ", len(dat.columns))
print("Number of rows : ", len(dat.index))


############# Correlation matrix ############
correlation_mat = dat.corr()
#print (correlation_mat)
sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of strains")

plt.xlabel("strains")
plt.ylabel("strains features")

plt.show()   
############# Correlation matrix ############


############# PCA - visualize projections - USER DEFINED ALGORITHM #1 - NOT SO GOOD - START ############ 


# n = len(dat.columns)

# pca = PCA(n_components = n)
# # defaults number of PCs to number of columns in imported data (ie number of
# # features), but can be set to any integer less than or equal to that value

# pca.fit(dat)


# ## project data into PC space

# # 0,1 denote PC1 and PC2; change values for other PCs
# xvector = pca.components_[0] # see 'prcomp(my_data)$rotation' in R
# yvector = pca.components_[1]
# zvector = pca.components_[2]

# xs = pca.transform(dat)[:,0] # see 'prcomp(my_data)$x' in R
# ys = pca.transform(dat)[:,1]
# zs = pca.transform(dat)[:,2]


    
# ## Note: scale values for arrows and text are a bit inelegant as of now,
# ##       so feel free to play around with them

# for i in range(len(xvector)):
# # arrows project features (ie columns from csv) as vectors onto PC axes
#     plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
#               color='r', width=0.0005, head_width=0.0025)
#     plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
#               list(dat.columns.values)[i], color='r')

# for i in range(len(xs)):
# # circles project documents (ie rows from csv) as points onto PC axes
#     plt.plot(xs[i], ys[i], 'bo')
#     plt.text(xs[i]*1.2, ys[i]*1.2, list(dat.index)[i], color='b')

# plt.show()


############# PCA - visualize projections - USER DEFINED ALGORITHM #1 - NOT SO GOOD - END ############ 


############# PCA - visualize projections - USER DEFINED ALGORITHM #2 - NOT SO GOOD - START ############ 

# #In general a good idea is to scale the data
# scaler = StandardScaler()
# scaler.fit(dat)
# X=scaler.transform(dat)    

# pca = PCA()
# x_new = pca.fit_transform(dat)

# def myplot(score,coeff,labels=None):
#     xs = score[:,0]
#     ys = score[:,1]
#     n = coeff.shape[0]
#     scalex = 1.0/(xs.max() - xs.min())
#     scaley = 1.0/(ys.max() - ys.min())
#     plt.scatter(xs * scalex,ys * scaley)
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
#         if labels is None:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
#         else:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
            
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.xlabel("PC{}".format(1))
# plt.ylabel("PC{}".format(2))
# plt.grid()

# #Call the function. Use only the 2 PCs.
# myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
# plt.show()


############# PCA - visualize projections - USER DEFINED ALGORITHM #2 - NOT SO GOOD - END ############ 



############# PCA Biplot - visualize projections - Library  ALGORITHM - VERY GOOD - Start ############ 

print("\n\nPCA BiPlo - Process .....")
# Import libraries
from pca import pca

font = {'family' : 'italic',
        'weight' : 'bold',
        'size'   : 22}

###########  User Defined Data ##############################
# # Lets create a dataset with features that have decreasing variance. 
# # We want to extract feature f1 as most important, followed by f2 etc
# f1=np.random.randint(0,100,250)
# f2=np.random.randint(0,50,250)
# f3=np.random.randint(0,25,250)
# f4=np.random.randint(0,10,250)
# f5=np.random.randint(0,5,250)
# f6=np.random.randint(0,4,250)
# f7=np.random.randint(0,3,250)
# f8=np.random.randint(0,2,250)
# f9=np.random.randint(0,1,250)

# # Combine into dataframe
# X = np.c_[f1,f2,f3,f4,f5,f6,f7,f8,f9]
# X = pd.DataFrame(data=X, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9'])

###########  User Defined Data ##############################

#scaler = StandardScaler()
#scaler.fit(dat)
#dat=scaler.transform(dat)    

# Initialize
model = pca(n_components=0.95, n_feat=len(dat.index), alpha=0.05, n_std=2, onehot=False, normalize=False, detect_outliers=['ht2','spe'], random_state=None)

# Fit transform
out = model.fit_transform(dat)


# Print the top features. The results show that f1 is best, followed by f2 etc
#print("\ntopfeat: \n",  out['topfeat'])
#print("\noutliers:\n",  out['outliers'])
#print("\nshape : \n",   dat.shape)


fig, ax = model.plot(figsize=(20,18))
fig, ax = model.biplot(cmap=None, figsize=(20,18), label=False, legend=True)

# Create only the scatter plots
# print("\n\n\nCreate only the scatter plots")
fig, ax = model.scatter(figsize=(20,18),legend=True, cmap='Set2', label=False)
fig, ax = model.scatter3d(figsize=(20,18), legend=True, cmap='Set2', label=False)


# Create only the bi plots
print("\n\n\nCreate only the bi plots")
fig, ax = model.biplot(n_feat=len(dat.index), figsize=(20,18), legend=True, cmap='Set2', label = False)
fig, ax = model.biplot3d(n_feat=len(dat.index), figsize=(20,18), legend=True,cmap='Set2', label = False)


############# PCA Biplot - visualize projections - Library  ALGORITHM - VERY GOOD - End ############ 

