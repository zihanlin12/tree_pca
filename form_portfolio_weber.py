# Use the adaptive lasso method as that in weber's paper.
# Use ADMM

import argparse
import os
import pandas as pd
import numpy as np
from sklearn import linear_model

parser = argparse.ArgumentParser(description='')
parser.add_argument('--J', default=2, help='How many to pick for each characteristics')
parser.add_argument('--M', default= 3, help= 'The order of characteristics')
parser.add_argument('--tic', default='Return,BMclean,MOMclean,Sizeclean,vol', help= 'All of the characteristics')
parser.add_argument('--epsilon', default= 1e-2, help= 'The threshold for convergence.')
args = parser.parse_args()

File_name= args.tic.split(',')
# print (File_name)
# One problem is that the final column is missing.

# Not using all of the characteristics

# Sharpe is too high, unexpected

# The second round wrong, and there are problems with beta_2.

ret= File_name[0]
chara= File_name[1:]

J= args.J
M= args.M
window= 120
epsilon= args.epsilon

# T= 360
# N= 720
# L= len(File_name)-1
# print (L)
# chara_data= np.zeros((T, N, L))

# file_msg = './cleaned_data/%s.csv' % (ret)
# df = pd.read_csv(file_msg, header= None)
# returns = pd.DataFrame.as_matrix(df)

# for i, file in enumerate(chara):
# 	# print (file)
# 	file_msg = './cleaned_data/%s.csv' % (file)
# 	df = pd.read_csv(file_msg, header= None)		
# 	data= pd.DataFrame.as_matrix(df)
# 	chara_data[:,:,i]= data

# data= np.load('./npz_data/Np_processed.npz')
# data= data['data']
# T, N, L= data.shape
# T= T+1
# returns= np.zeros((T, N))
# returns[1:]= data[:,:,0]
# index= [1]
# index+= list(range(5,L))
# chara_data= data[:,:,index]
# L= chara_data.shape[2]

# fama_french_data= np.load('./npz_data/riskfreerate.npz')
# fama_french_data= fama_french_data['fama_french_data']
# r_f= fama_french_data[:,4]
# fama_french_date= fama_french_data[:,0]
# r_f= r_f/100
# date_start= np.asscalar(np.argwhere(fama_french_date==198401))
# date_end= np.asscalar(np.argwhere(fama_french_date==201401))
# returns= returns-r_f[date_start:date_end:,np.newaxis]

data= np.load('./npz_data/processed_data.npz')
data= data['data']
T, N, L= data.shape
T= T+1
returns= np.zeros((T, N))
returns[1:]= data[:,:,0]
index= [1]
index+= list(range(5,L))
chara_data= data[:,:,index]
L= chara_data.shape[2]

fama_french_data= np.load('./npz_data/riskfreerate.npz')
fama_french_data= fama_french_data['fama_french_data']
r_f= fama_french_data[:,4]
fama_french_date= fama_french_data[:,0]
r_f= r_f/100
date_start= np.asscalar(np.argwhere(fama_french_date==198401))
date_end= np.asscalar(np.argwhere(fama_french_date==201401))
returns= returns-r_f[date_start:date_end:,np.newaxis]

portfolio= np.zeros((T-1, N))

def group_lasso(Y, X, lambda_1):
	K= X.shape[1]
	beta_2= np.random.randn(K,1)
	beta_1= np.ones((K,1))
	while np.linalg.norm(beta_1-beta_2)>epsilon:
		beta_1= beta_2
		for i in range(L):
			X_j= X[:, i*(J+M):(i+1)*(J+M)]
			S_j= X_j.T.dot(Y- X.dot(beta_1)+ X_j.dot(beta_1[i*(J+M):(i+1)*(J+M)]))
			beta_2[i*(J+M):(i+1)*(J+M)]= np.linalg.solve(X_j.T.dot(X_j), np.maximum(1-lambda_1/np.linalg.norm(S_j), 0)*S_j)
	return beta_2

def second_group_lasso(Y, X, beta, lambda_2):
	length= X.shape[1]
	index= []
	for i in range(L):
		beta_i = beta[i*(J+M):(i+1)*(J+M)]
		if np.linalg.norm(beta_i)!=0:
			index+= list(range(i*(J+M),(i+1)*(J+M)))
	X= X[:, index]
	K= X.shape[1]
	L_prime= int(K/(J+M))
	beta_2= np.random.randn(K,1)
	beta_1= np.ones((K, 1))
	# I need to get the omegas.
	while np.linalg.norm(beta_1-beta_2)>epsilon:
		beta_1= beta_2
		for i in range(L_prime):
			beta_i= beta[index[i*(J+M):(i+1)*(J+M)]]
			w_t= 1/np.linalg.norm(beta_i)
			X_j= X[:, i*(J+M):(i+1)*(J+M)]
			S_j= X_j.T.dot(Y- X.dot(beta_1)+ X_j.dot(beta_1[i*(J+M):(i+1)*(J+M)]))
			beta_2[i*(J+M):(i+1)*(J+M)]= np.linalg.solve(X_j.T.dot(X_j), np.maximum(1-lambda_2*w_t/np.linalg.norm(S_j), 0)*S_j)
	beta_output= np.zeros((K, 1))
	beta_output[index, :]= beta_2
	return beta_output
	
def tuning(Y, X, lambda_list):
	BIC= np.zeros(len(lambda_list))
	beta= []
	for i, lambda_1 in enumerate(lambda_list):
		beta.append(group_lasso(Y, X, lambda_1)) 
		BIC[i]= get_BIC(Y, X, beta[i])
	return lambda_list[np.argmin(BIC)], beta[np.argmin(BIC)]

def second_tuning(Y, X, lambda_list, beta_1):
	BIC= np.zeros(len(lambda_list))
	beta= []
	for i, lambda_1 in enumerate(lambda_list):
		if i==len(lambda_list)-1:
			print ('Let see what is going on here')
		beta.append(second_group_lasso(Y, X, beta_1,  lambda_1))
		BIC[i]= get_BIC(Y, X, beta[i])
	return lambda_list[np.argmin(BIC)], beta[np.argmin(BIC)]

def get_BIC(Y, X, beta):
	regr= linear_model.LinearRegression()
	regr.fit(X, Y)
	beta_LS= regr.coef_
	T= X.shape[0]
	U= Y- X.dot(beta)
	sigma_2= U.T.dot(U)/(T-1)
	df= 0
	for i in range(L):
		beta_j= beta[i*(J+M):(i+1)*(J+M)]
		df+= 1 if np.linalg.norm(beta_j)>0 else 0
		beta_j_LS= beta_LS[:,i*(J+M):(i+1)*(J+M)]
		if np.linalg.norm(beta_j_LS)==0:
			print ('What is going on?')
		df+= np.linalg.norm(beta_j)/np.linalg.norm(beta_j_LS)*(J+M-2)
	return (U.T.dot(U)/sigma_2-T+2*df)

def form_optimal(returns, chara, beta):
	# form a long short portfolio based on the predicted return, and find the 
	predicted_return= form_basis(chara).dot(beta)
	predicted_median= np.median(predicted_return)
	ones= np.ones(returns.shape)
	for i in range(len(ones)):
		if predicted_return[i] < predicted_median:
			ones[i]= -1
	ones/= len(ones)
	# ones[predicted_return<predicted_median]= -1
	return np.mean(ones*returns)


def form_basis(chara_data):
	basis= np.ones((N, (J+M)*L))
	for j in range(L):
		sample= chara_data[:,j]
		quantile= np.array([np.percentile(sample, int(100.0/(J+1)*(k))) for k in range(1,J+1)])
		basis[:,j*(J+M):(j*(J+M)+J)]= np.maximum(np.tile(sample[:,np.newaxis], J)-np.tile(quantile[:, np.newaxis], N).T, 0)
		basis[:,j*(J+M):(j*(J+M)+J)]= np.power(basis[:,j*(J+M):(j*(J+M)+J)] , M-1)
		basis[:,(j*(J+M)+J):(j*(J+M)+J+M)]=  np.stack([np.power(sample, k) for k in range(0, M)], axis= 1)
	return basis

def get_sharpe(optimal_return):
	return np.mean(optimal_return)/np.sqrt(np.cov(optimal_return))

def normalize_chara(chara_data):
	N= chara_data.shape[1]
	chara_data= np.argsort(chara_data, axis= 1)/(N+1)
	return chara_data

optimal_return= np.zeros((T-1-window))
chara_data= normalize_chara(chara_data)
for i in range(T-1-window):
	print ('I am currently at %d'%(i))
	chara_data_i= chara_data[i:(i+window),:,:]
	basis= np.ones((window, N, (J+M)*L))
	for j in range(L):
		sample= chara_data_i[:,:,j]
		# print (sample.shape)
		quantile= np.stack([np.percentile(sample, int(100.0/(J+1)*(k)), axis= 1) for k in range(1,J+1)], axis= 1)
		# print (quantile.shape)
		basis[:,:,j*(J+M):(j*(J+M)+J)]= np.maximum(np.tile(sample[:,:,np.newaxis], [1,1,J])-np.tile(np.expand_dims(quantile,1), [1,N,1]), 0)
		basis[:,:,j*(J+M):(j*(J+M)+J)]= np.power(basis[:,:,j*(J+M):(j*(J+M)+J)] , M-1)
		basis[:,:,(j*(J+M)+J):(j*(J+M)+J+M)]=  np.stack([np.power(sample, k) for k in range(0, M)], axis= 2)
	# basis[:,:,-1]= np.power(sample,0)
	Y= np.reshape(returns[(i+1):(i+window+1),:],(-1,1))
	X= np.reshape(basis, (-1,(J+M)*L))
	lambda_list= [0.1,1,10,100]
	lambda_1, beta = tuning(Y, X, lambda_list)
	lambda_2, beta= second_tuning(Y, X, lambda_list, beta)
	optimal_return[i]= form_optimal(returns[i+window+1,:], chara_data[(i+window),:], beta)
print ('Just for debug')
print (get_sharpe(optimal_return))

