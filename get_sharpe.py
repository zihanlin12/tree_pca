import pandas as pd
import numpy as np
import os
import scipy.linalg as la
import argparse

parser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('--RP', default=1, help='Whether use Risk Premium PCA')
parser.add_argument('--Corr', default= 1, help = 'Whether use correlation matrix')
parser.add_argument('--Time', default= 1, help= 'Whether use time dimension or not')
parser.add_argument('--delta', default= 600, help = 'The number of portfolios left in each factor')
parser.add_argument('--Shrink', default= 0, help = 'Whether you want to apply the proxy factor or not')
args = parser.parse_args()

current_dir= os.getcwd()
data_dir= current_dir+'/npz_data/simulated_portfolio.npz'
# data_dir= current_dir+'/npz_data/Fan_portfolio.npz'
# data_dir= current_dir+'/npz_data/simulated_decile_portfolio.npz'
data= np.load(data_dir)
portfolio= data['portfolio']
def compute_sharpe(K, portfolio, RP= True, Corr= True, Time= True, delta= 600, Shrink= True):
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
            optimal_return[t] = (portfolio[t + window, :].dot(loading_t)).dot(np.linalg.inv(loading_t.T.dot(loading_t))).dot(weight)
        else:
            optimal_return[t] = (portfolio[t + window, :].dot(loading_t)).dot(np.linalg.inv(loading_t.T.dot(loading_t))).dot(weight)
    return optimal_return
def easy_sharpe(optimal_return):
        mean_optimal= np.mean(optimal_return)
        cov_optimal= np.cov(optimal_return)
        return abs(mean_optimal)/np.sqrt(cov_optimal)
K= 5
optimal_sharpe= np.zeros((K))
for i in range(K):
    print (args.Shrink)
    optimal_sharpe[i]= easy_sharpe(compute_sharpe(i+1, portfolio, int(args.RP), int(args.Corr), int(args.Time), int(args.delta), int(args.Shrink)))
print (optimal_sharpe)
