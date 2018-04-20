# This file implements the individual stock of 
import pandas as pd
import numpy as np
import argparse
import copy
import scipy.linalg as la
import random

parser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('--start_time',default= 198401, help= 'Please type in the starting time')
parser.add_argument('--end_time', default= 201401, help= 'Please type in the ending time')
parser.add_argument('--RP', default=1, help='Whether use Risk Premium PCA')
parser.add_argument('--Corr', default= 1, help = 'Whether use correlation matrix')
parser.add_argument('--Time', default= 1, help= 'Whether use time dimension or not')
parser.add_argument('--delta', default= 300, help = 'The number of portfolios left in each factor')
parser.add_argument('--Shrink', default= 0, help = 'Whether you want to apply the proxy factor or not')
args = parser.parse_args()

file_msg= './Jason_data/RET.csv'
file_data= pd.read_csv(file_msg)
dates= file_data['Date']
dates= dates//100
dates_list= list(dates)
time_idx= {date:i for i, date in enumerate(dates_list)}
idx_start = time_idx[args.start_time]
idx_end= time_idx[args.end_time]
window= 120
firm_name= file_data.axes[1][1:]
possible_name= [i for i in range(len(firm_name))]

firm_index= []

for i in range(idx_start+window, idx_end):
	possible_name_copy= [k for k in range(len(firm_name))]
	for j in range(i-window, i+1):
		# print (j== idx_start)
		data_i= file_data.iloc[j+1][1:]
		possible_name_temp= [k for k in possible_name_copy if not np.isnan(data_i[k])]
		possible_name_copy= possible_name_temp
	# possible_name_copy= [k+1 for k in possible_name_copy]
	# portfolio_t= file_data.iloc[(i-window+1):(i+2),possible_name_copy]
	# print (portfolio_t.shape)
	# print (np.sum(np.isnan(pd.DataFrame.as_matrix(portfolio_t))))
	firm_index.append(possible_name_copy)

portfolio= pd.DataFrame.as_matrix(file_data.iloc[idx_start+1:(idx_end+1),1:])


def compute_sharpe(K, portfolio, firm_index,  RP= True, Corr= True, Time= True, delta= 600, Shrink= True):
	T, M = portfolio.shape
	window= 120
	optimal_return = np.zeros((T-window))
	gamma = 10    
	ones_T= np.ones((window, 1))
	M= min(1000, min([len(index) for index in firm_index]))
	print (M)
	for t in range(T-window):
		print (t)
		firm_index[t]= random.sample(firm_index[t], M)
		portfolio_t = portfolio[t:(t + window), firm_index[t]]
		print ('The number of nan elements in this range is %d'%(np.sum(np.isnan(portfolio_t))))
		if Time: 
			if RP:
				covmat_t= (np.identity(window)+ones_T.dot(ones_T.T)/(window*(2+window))).dot(portfolio_t)
				covmat_t= (covmat_t.dot(portfolio_t.T)).dot(np.identity(window)+ones_T.dot(ones_T.T)/(window*(2+window)))       
			else: 
				covmat_t= np.cov(portfolio_t)        
		else:
			if RP:
				covmat_t = portfolio_t.T.dot(portfolio_t) 
				portfolio_mean_t = np.sum(portfolio_t, axis=0, keepdims= True)
				covmat_t = covmat_t + portfolio_mean_t.T.dot(portfolio_mean_t)/window * gamma
			else:
				covmat_t= np.cov(portfolio_t.T)
		if Corr:
			covmat_t_diag= np.diag(1./np.sqrt(np.diag(covmat_t)))
			covmat_t= covmat_t_diag.dot(covmat_t.dot(covmat_t_diag))
		if Time:
			variance_t, Factor_t= la.eigh(covmat_t, eigvals=(window-K, window-1))
			Factor_t= Factor_t[:,:K]
			loading_t= Factor_t.T.dot(portfolio_t)
			loading_t= loading_t.T
		else:
			variance_t, loading_t= la.eigh(covmat_t, eigvals=(M-K, M-1))
			Factor_t= portfolio_t.dot(loading_t)
		if Shrink:
			for i in range(K):
				threshold= np.partition(abs(loading_t[:,i]),-delta)[-delta]
				loading_t[np.abs(loading_t[:,i])<threshold,i]=0
				loading_t[:,i]= loading_t[:,i]/np.linalg.norm(loading_t[:,i])
			Factor_t= portfolio_t.dot(loading_t).dot(np.linalg.inv(loading_t.T.dot(loading_t)))
				
		mu= np.mean(Factor_t, axis= 0)
		covmat_factor= np.cov(Factor_t.T)
		if K>=2:
			weight= np.linalg.inv(covmat_factor).dot(mu)
		else:
			weight= mu/covmat_factor
		if Time:        
			optimal_return[t] = (portfolio[t + window, firm_index[t]].dot(loading_t)).dot(np.linalg.inv(loading_t.T.dot(loading_t))).dot(weight)
		else:
			optimal_return[t] = (portfolio[t + window, firm_index[t]].dot(loading_t)).dot(np.linalg.inv(loading_t.T.dot(loading_t))).dot(weight)
	return optimal_return
def easy_sharpe(optimal_return):
	mean_optimal= np.mean(optimal_return)
	cov_optimal= np.cov(optimal_return)
	return abs(mean_optimal)/np.sqrt(cov_optimal)
K= 10
optimal_sharpe= np.zeros((K))
for i in range(K):
	optimal_sharpe[i]= easy_sharpe(compute_sharpe(i+1, portfolio, firm_index,  int(args.RP), int(args.Corr), int(args.Time), int(args.delta), int(args.Shrink)))
print (optimal_sharpe)


