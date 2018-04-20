import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la



def stable_loading(window, K, portfolio_all, RP= True, Corr= True, Time= True, delta= 600, Shrink= True):
	T, M= portfolio_all.shape
	for t in range(T-window):
		portfolio_t= portfolio_all[t:t+window,:]
		loading_t= get_loading(K, portfolio_t, RP, Corr, Time, delta, Shrink)
		base= 230
		for k in range(6):
			id= base+k+1
			plt.subplot(id)
			plt.plot(loading_t)
	print ('I have been here')
	plt.show()

def get_loading(K, portfolio, RP= True, Corr= True, Time= True, delta= 600, Shrink= True):
	T, M= portfolio.shape
	gamma= 10
	ones_T= np.ones((T,1))
	if Time: 
		if RP:
			covmat_t= (np.identity(T)+ones_T.dot(ones_T.T)/(T*(2+T))).dot(portfolio)
			covmat_t= (covmat_t.dot(portfolio.T)).dot(np.identity(T)+ones_T.dot(ones_T.T)/(T*(2+T)))       
		else: 
			covmat_t= np.cov(portfolio)        
	else:
		if RP:
			covmat_t = portfolio.T.dot(portfolio) 
			portfolio_mean_t = np.sum(portfolio, axis=0, keepdims= True)
			covmat_t = covmat_t + portfolio_mean_t.T.dot(portfolio_mean_t)/T * gamma
		else:
			covmat_t= np.cov(portfolio.T)
	if Corr:
		covmat_t_diag= np.diag(1./np.sqrt(np.diag(covmat_t)))
		covmat_t= covmat_t_diag.dot(covmat_t.dot(covmat_t_diag))
	if Time:
		variance_t, Factor_t= la.eigh(covmat_t, eigvals=(T-K, T-1))
		Factor_t= Factor_t[:,:K]
		loading_t= Factor_t.T.dot(portfolio)
		loading_t= loading_t.T
	else:
		variance_t, loading_t= la.eigh(covmat_t, eigvals=(M-K, M-1))
	if Shrink:
		for i in range(K):
			threshold= np.partition(abs(loading_t[:,i]),-delta)[-delta]
			loading_t[np.abs(loading_t[:,i])<threshold,i]=0
			loading_t[:,i]= loading_t[:,i]/np.linalg.norm(loading_t[:,i])
	return loading_t

portfolio_all= np.load('./npz_data/tree_portfolio.npz')
portfolio_all= portfolio_all['portfolio']
stable_loading(320, 6, portfolio_all, RP= True, Corr= True, Time= True, delta= 600, Shrink= True)