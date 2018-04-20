import numpy as np
import math
from scipy.stats import norm
import os

current_dir= os.getcwd()
data_dir= current_dir+'/npz_data_balanced/tree_portfolio.npz'
data= np.load(data_dir)
portfolio_tree= data['portfolio']
data_dir= current_dir+'/npz_data_balanced/decile_portfolio.npz'
data= np.load(data_dir)
portfolio_decile= data['portfolio']
def compute_alpha(K,portfolio,portfolio_test, RP= True, Corr= True, Time= True, delta= 600, Shrink= True):
    # portfolio is the portfolio from which we get the factors, and portfolio_test is where we would like to test these factors.
    import scipy.linalg as la
    T, M = portfolio_test.shape
    gamma = 10  
    pricing_error= np.zeros((T,M))
    ones_T= np.ones((T, 1))
    portfolio_t = portfolio
    if Time: 
        if RP:
            covmat_t= (np.identity(T)+ones_T.dot(ones_T.T)/(T*(2+T))).dot(portfolio_t)
            covmat_t= (covmat_t.dot(portfolio_t.T)).dot(np.identity(T)+ones_T.dot(ones_T.T)/(T*(2+T)))       
        else: 
            covmat_t= np.cov(portfolio_t)        
    else:
        if RP:
            covmat_t = portfolio_t.T.dot(portfolio_t) 
            portfolio_mean_t = np.sum(portfolio_t, axis=0, keepdims= True)
            covmat_t = covmat_t + portfolio_mean_t.T.dot(portfolio_mean_t)/T * gamma
        else:
            covmat_t= np.cov(portfolio_t.T)
    if Corr:
        covmat_t_diag= np.diag(1./np.sqrt(np.diag(covmat_t)))
        covmat_t= covmat_t_diag.dot(covmat_t.dot(covmat_t_diag))
        # Get the correlation matrix
    if Time:
        variance_t, Factor_t= la.eigh(covmat_t, eigvals=(T-K, T-1))
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
        Factor_t= portfolio_t.dot(loading_t)
    portfolio_e= portfolio_test
    loading_test= portfolio_e.T.dot(Factor_t).dot(np.linalg.inv(Factor_t.T.dot(Factor_t)))
    pricing_error = (portfolio_test-Factor_t.dot(loading_test.T))
    M_F= np.identity(T)-Factor_t.dot(np.linalg.inv(Factor_t.T.dot(Factor_t))).dot(Factor_t.T)
    a_F_T= (np.ones((1,T)).dot(M_F).dot(np.ones((T,1))))
    avg_pricing_error= np.sum(pricing_error, axis= 0)/a_F_T
    return avg_pricing_error, pricing_error, a_F_T
portfolio_tree= portfolio_tree*100
portfolio_decile= portfolio_decile*100
pricing_error, U, a_F_T= compute_alpha(4, portfolio_decile, portfolio_tree, True, True, True, 10, False)
print (U)
def get_covariance(U, K):
    T,M= U.shape
    Sigma= 1/T*U.T.dot(U)
    temp_matrix= np.tile(np.expand_dims(U, axis= 2), [1, 1, M])*np.tile(np.expand_dims(U, axis= 1), [1, M, 1])-np.tile(np.expand_dims(Sigma,0),[T,1,1]) 
    hat_theta= np.mean(temp_matrix**2, axis= 0)
    w_T= 0.010*K*math.sqrt(math.log1p(M)/T)
    print (w_T)
    print (np.max(Sigma))
    index= (Sigma>np.sqrt(hat_theta)*w_T).astype(int)
    Sigma_u= Sigma*index
    print (np.linalg.matrix_rank(Sigma_u))
    return Sigma_u
Sigma_u= get_covariance(U, 4)

# Final testing
T,M= U.shape
output= (a_F_T*pricing_error.dot(np.linalg.solve(Sigma_u, pricing_error.T))-M)/sqrt(2*M)
print 2*(1-norm.cdf(output/2))

