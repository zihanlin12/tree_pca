import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

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

def beautiful_print(alist):
    char_list= ['momentum', 'size', 'Idio_vol', 'beme','Book']
    size_list= ['small', 'big']
    out_list= []
    for char in alist:
        out_list.append((char_list[char[0]], size_list[char[1]]))
    return out_list

portfolio_all= np.load('./npz_data/large_tree_portfolio.npz')
portfolio= portfolio_all['portfolio']
nodetag= portfolio_all['nodetag']
loading= get_loading(6, portfolio, RP= True, Corr= True, Time= True, delta= 600, Shrink= False)
delta= 10
count= 0
for i in range(1):
    print ("This is the %d factor" % i)
    threshold= np.partition(abs(loading[:,i]),-delta)[-delta]
    # loading[np.abs(loading[:,i])<threshold,i]=0
    # loading_2_K[:,i]= loading_2_K[:,i]/np.linalg.norm(loading_2_K[:,i])*np.sqrt(N)
    index_K= np.argwhere(np.abs(loading[:,i])>=threshold)
    for j in range(len(index_K)):
        print(beautiful_print(nodetag[index_K[j][0]]))
        print (loading[j, i])
        count+= 1