# See which does better
from sklearn import linear_model
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('k', default=5, help='The range of alpha to choose from')
args = parser.parse_args()

class Lasso_model:
	def __init__(self, portfolio, k):
		self.portfolio= portfolio
		self.k= k

	def cross_validation(self):
		alpha= [2**(i) for i in np.arange(-self.k, self.k)]
		out_sample_sharpe= 

	def compute_sharpe(self, K , portfolio , RP= True, Corr= True, Time= True, delta= 600, Shrink= True):
    	import scipy.linalg as la
    	T, M = portfolio.shape
    	window= 120
    	optimal_return = np.zeros((T-window))
    	gamma = 10    
    	ones_T= np.ones((window, 1))
    	for t in range(T-window):
        	portfolio_t = portfolio[t:(t + window), :]
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
        # Get the correlation matrix
        	if Time:
            	variance_t, Factor_t= la.eigh(covmat_t, eigvals=(window-K, window-1))
            	Factor_t= Factor_t[:,:K]
            	loading_t= Factor_t.T.dot(portfolio_t)
        	else:
            	variance_t, loading_t= la.eigh(covmat_t, eigvals=(M-K, M-1))
            	Factor_t= portfolio_t.dot(loading_t)
        	if Shrink:
            	for i in range(K):
                	threshold= np.partition(abs(loading_t[:,i]),-delta)[-delta]
                	loading_t[np.abs(loading_t[:,i])<threshold,i]=0
                	loading_t[:,i]= loading_t[:,i]/np.linalg.norm(loading_t[:,i])
                
        	mu= np.mean(Factor_t, axis= 0)
        	covmat_factor= np.cov(Factor_t.T)
        	if K>=2:
            	weight= np.linalg.inv(covmat_factor).dot(mu)
        	else:
            	weight= mu/covmat_factor
        	if Time:        
            	optimal_return[t] = (portfolio[t + window, :].dot(loading_t.T)).dot(np.linalg.inv(loading_t.dot(loading_t.T))).dot(weight)
        	else:
            	optimal_return[t] = (portfolio[t + window, :].dot(loading_t)).dot(weight)
    	return optimal_return
	def easy_sharpe(self, optimal_return):
        mean_optimal= np.mean(optimal_return)
        cov_optimal= np.cov(optimal_return)
        return abs(mean_optimal)/np.sqrt(cov_optimal)


if __name__== '__main__':
	portfolio= np.load('./npz_data/tree_portfolio.npz')
	Lasso_model= Lasso_model(portfolio, args.k)
