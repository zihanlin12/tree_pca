import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la


real_factor= np.load('./npz_data/simulated_factor.npz')
real_factor= real_factor['factor']

tree_portfolio= np.load('./npz_data/simulated_portfolio.npz')
tree_portfolio= tree_portfolio['portfolio']
decile_portfolio= np.load('./npz_data/simulated_decile_portfolio.npz')
decile_portfolio= decile_portfolio['portfolio']

def get_factor(K, portfolio, RP= True, Corr= True, Time= True, delta= 600, Shrink= True):
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
		Factor_t= portfolio.dot(loading_t)
	if Shrink:
		for i in range(K):
			threshold= np.partition(abs(loading_t[:,i]),-delta)[-delta]
			loading_t[np.abs(loading_t[:,i])<threshold,i]=0
			loading_t[:,i]= loading_t[:,i]/np.linalg.norm(loading_t[:,i])
		Factor_t= portfolio.dot(loading_t).dot(np.linalg.inv(loading_t.T.dot(loading_t)))
	return Factor_t

base= 220

Factor_decile= get_factor(4, decile_portfolio, True, True, 600, False)
Factor_tree= get_factor(4, tree_portfolio, True, True, 600, False)
for k in range(4):
	id= base+k+1
	plt.subplot(id)
	factor_true= real_factor[:,k]
	factor_decile= Factor_decile[:,k]
	factor_tree= Factor_tree[:,k]
	plt.plot(np.cumsum(factor_decile), label= 'Decile factor')
	plt.plot(np.cumsum(factor_tree), label= 'Tree factor')
	plt.plot(np.cumsum(factor_true), label= 'True factor')
	plt.title('The %d th factor'%(k+1))
	if k==0:
		plt.legend(loc=3)
		# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
  	
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()


