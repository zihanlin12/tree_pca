import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import scipy.linalg as la


parser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('--mode', default= 0, help='0 means the Weber data, 1 means Markus data, 2 means processed data of my own')
parser.add_argument('--use_corr', default= 'True', help='Whether you want to use correlation or covariance matrix')
parser.add_argument('--portfolio', default= 'Tree', help= 'Which sorting strategy you want to use? Tree, Decile, or Fama French?')
parser.add_argument('--delta', default= 600, help= 'How much non zero asset to keep in each portfolio')
parser.add_argument('--shrink', default= 'True', help= 'Whether I want to shrink the loading or not.')
parser.add_argument('--Time', default=True, help= 'Whether you are using time series time dimension or not')
parser.add_argument('--RP', default= True, help= 'Whether you want to use RPPCA or not.')



args = parser.parse_args()

def read_data(args):
	if args.mode==0:
		print ('I have been here')
		data = np.load('./npz_data/bryankelly_without_missing.npz')
		processed_data = data['processed_data']
		T, N, M = processed_data.shape
		returns = processed_data[:, :, 1]
		fama_french_data= np.load('./npz_data/riskfreerate.npz')
		fama_french_data= fama_french_data['fama_french_data']
		r_f= fama_french_data[:,4]
		r_f= r_f/100
		date= fama_french_data[:,0]
		date_start= np.asscalar(np.argwhere(date==198406))
		date_end= np.asscalar(np.argwhere(date==201405))
		returns= returns- (r_f[date_start:date_end+1:,np.newaxis])
	# The dimension of returns: T*N
		chara_data = processed_data[:, :, 5:9]
		return returns, chara_data
	elif args.mode==1:
		data= np.load('./')
	elif args.mode==2:
		data= np.load('./')
	


def get_tree(depth, res, nodetag,  value_array, L, d, chara_data, oldlist, parent):
	L, N= chara_data.shape	
	if depth == d:
		final_array = np.ones([N, 1])
		for k in range(L):
			temp = value_array[:, k].reshape([N, 1])
			final_array = final_array * temp
		if np.sum(final_array) < 10 ** (-3):
			print('An error over here')
		final_array = 1.0 * final_array / np.sum(final_array)
		res.append([final_array]) 
		nodetag.append(oldlist)
	else:
		for i in range(L):
			for j in range(2):
				if len(parent) != 0:
					index = np.ones((N), dtype=bool)
					for k in parent:
						index = (index) & ((value_array[:, k] != 0))
					data_median = np.median(chara_data[i, index])
				else:
					index = (value_array[:, i] != 0)
					data_median = np.median(chara_data[i, index])
				temp_array = value_array.copy()
				newlist= list(oldlist)
				newlist+=[(i,j)]
				parent.append(i)
				if j == 0:
					temp_array[(index) & (chara_data[i, :] <= data_median), i] = 0
					print (temp_array==value_array)
					get_tree(depth + 1, res, nodetag, temp_array, L, d, chara_data, newlist, parent)
				elif j == 1:
					temp_array[(index) & (chara_data[i, :] >= data_median), i] = 0
					get_tree(depth + 1, res, nodetag, temp_array, L, d, chara_data, newlist, parent)
				parent.pop()

def form_portfolio(returns, chara_data):
	T, N, L= chara_data.shape
	d= 4
	for i in range(T-1):
		if i%20==0:
			print ('Have gone through %d' %(i))
		res = []
		nodetag= []
		portfolio= np.zeros(((2*L)**d))
		get_tree(0, res, nodetag, np.ones([N, L]), L, d, chara_data[i,:].T, [], [])
		for j in range(len(res)):
			portfolio[j]= returns[i+1,:].dot(res[j])
		path= './npz_data/portfolio%d.csv' % (i)
		np.savez(path, portfolio= portfolio)

def get_corr(start, end):
	length= end- start
	covmat_t= np.zeros((length, length))
	if not os.path.isfile('./npz_data/portfolio0.csv.npz'):
		form_portfolio(returns, chara_data)
	for i in range(length):
		for j in range(length):
			path_i= './npz_data/portfolio%d.csv.npz' % (start+i)
			portfolio_i= np.load(path_i)
			portfolio_i= portfolio_i['portfolio']
			path_j= './npz_data/portfolio%d.csv.npz' % (start+i)
			portfolio_j= np.load(path_j)
			portfolio_j= portfolio_j['portfolio']
			covmat_t[i,j]= np.dot(portfolio_i, portfolio_j)
	ones_T= np.ones((length, 1))
	auxi_mat= np.identity(length)+ones_T.dot(ones_T.T)/(length*(2+length))
	covmat_t= auxi_mat.dot(covmat_t.dot(auxi_mat))
	if args.use_corr:
		covmat_t_diag= np.diag(1./np.sqrt(np.diag(covmat_t)))
		covmat_t= covmat_t_diag.dot(covmat_t.dot(covmat_t_diag))
	return covmat_t


def out_of_RPPCA(K, chara_data, d):
	T, N, L= chara_data.shape
	print (T)
	M= (2*L)**d
	window= 120
	delta= args.delta
	optimal_return = np.zeros((T-window))
	gamma = 10   
	for t in range(T-window):	
		print ('I am at %d'%(t))	
		covmat_t= get_corr(t, t+window)
		variance_t, Factor_t= la.eigh(covmat_t, eigvals=(window-K, window-1))
		Factor_t= Factor_t[:,:K]
		loading_t=np.zeros((K,M))
		for i in range(M):
			# The dimension of loading_t is K*M			
			for j in range(window):
				path= './npz_data/portfolio%d.csv.npz'%(t+j)
				portfolio_t= np.load(path)
				portfolio_t= portfolio_t['portfolio']
				# print (portfolio_t.shape)
				# print (Factor_t.shape)
				loading_t[:,i] += Factor_t[j,:]*(portfolio_t[i])
				# K M           Factor: T*M   
		mu= np.mean(Factor_t, axis= 0)
		loading_t= loading_t.T
		covmat_factor= np.cov(Factor_t.T)
		if args.shrink:
			for i in range(K):
				threshold= np.partition(abs(loading_t[:,i]),-delta)[-delta]
				loading_t[np.abs(loading_t[:,i])<threshold,i]=0
				loading_t[:,i]= loading_t[:,i]/np.linalg.norm(loading_t[:,i])
		Factor_t= portfolio_t.dot(loading_t)
		if K>=2:     
			weight= np.linalg.inv(covmat_factor).dot(mu)
		else:
			weight= mu/covmat_factor
		path= './npz_data/portfolio%d.csv.npz'%(t+window)
		portfolio_out= np.load(path)
		portfolio_out= portfolio_out['portfolio']
		optimal_return[t] = (portfolio_out.dot(loading_t)).dot(np.linalg.inv(loading_t.T.dot(loading_t))).dot(weight)
	mean_optimal= np.mean(optimal_return)
	cov_optimal= np.cov(optimal_return)
	return (abs(mean_optimal)/np.sqrt(cov_optimal))



if __name__ == '__main__':
	return_data, chara_data= read_data(args)
	if not os.path.isfile('./npz_data/portfolio0.csv.npz'):
		form_portfolio(return_data, chara_data)
	print (out_of_RPPCA(3, chara_data, 4))