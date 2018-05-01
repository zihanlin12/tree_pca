import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data= pd.read_csv('25_Portfolios_ME_AC_5x5.csv')

data_for_pca= data.iloc[0:606,1:26]

X_std = StandardScaler().fit_transform(data_for_pca)
# X_std= data_for_pca
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Calculate the sharpe ratio of the factors
Factor = np.dot(data_for_pca, eig_vecs[:, 0:3])
mu = np.mean(Factor, axis=0)
cov_mat = np.cov(Factor.T)
max_sharpe = np.sqrt(np.dot(np.dot(mu, np.linalg.inv(cov_mat)), mu.T))

