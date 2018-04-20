# Perform the simulated estimator.

import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('--T', default=638, help='Whether use Risk Premium PCA')
parser.add_argument('--N', default= 1000, help = 'Whether use correlation matrix')
parser.add_argument('--M', default= 2, help = 'Whether use correlation matrix')
parser.add_argument('--rho', default= 0.7, help = 'Whether use correlation matrix')
parser.add_argument('--sigma_F_2', default= 0.03, help = 'Whether use correlation matrix')
parser.add_argument('--K', default= 4, help = 'Whether use correlation matrix')
parser.add_argument('--L', default= 3, help = 'The number of partitions in each coordinate')
parser.add_argument('--random', default= 1, help= 'Whether we are simulating the ')
args = parser.parse_args()

def get_correlation(N):
	Sigma= np.identity(N)
	assert N>=13, 'You should give a dimension that is larger'
	for i in range(13):
		Sigma[i+1, i]= 0.7
	Sigma= Sigma.dot(Sigma.T)
	Sigma= Sigma/np.diagonal(Sigma)
	return Sigma

def get_idiosyncratic(T, N):
	Sigma= get_correlation(N)
	e= np.random.randn(T, N).dot(Sigma)
	return e

def get_factor(T, K, sigma_F_2):
	# K is the number of factors
	diag= np.array([5, 0.3, 0.1, sigma_F_2])
	diag_F= np.sqrt(np.diag(diag))
	Factor= np.random.randn(T, K).dot(diag_F)
	Sharpe= np.array([0.12,0.1,0.3,0.5])
	print (np.mean(Factor, axis= 0))
	Factor= Factor+ Sharpe.dot(diag_F)
	return Factor

def get_characteristics(T, N, M, rho):
	# First of all, simulate the characteristics.
	base= np.random.randn(T, N, M)
	corre= np.diag(rho*np.ones((M-1)), 1)+ np.identity(M)
	# print (base.shape)
	# print (corre.shape)
	return np.matmul(base, corre)

def get_individual_loading(x, threshold, value):
	# Partition the whole domain into L 
	# Generate random numbers for the partition of domain and 
	M= len(x)
	K= threshold.shape[0]
	output= np.zeros((K))
	for i in range(K):
		threshold_i= threshold[i, :, :]
		# Dimension: K L M
		value_i= value[i,:]
		index= []
		for j in range(M):
			threshold_i_j= np.sort(threshold_i[:,j])
			index.append(get_index(threshold_i_j, x[j]))
		output[i]= value_i[tuple(index)]
	return output

def get_loading(characteristics, threshold, value):
	T, N, M= characteristics.shape
	K= threshold.shape[0]
	output= np.zeros((T, K, N))
	for i in range(T):
		for j in range(N):
			output[i,:,j]= get_individual_loading(characteristics[i,j,:], threshold, value)
		q, _= np.linalg.qr(output[i,:,:].T, mode= 'reduced')
		output[i,:,:]= q.T* np.sqrt(N)
	return output

def get_index(array, num):
	length= len(array)
	if array[0]>num:
		return 0
	if array[length-1]<num:
		return length-1
	for i in range(length):
		if i!= 0 and array[i]>num and array[i-1]<=num:
			return i

def get_partition(K, M, L):
	threshold= np.random.randn(K, L, M)
	my_list= [K]
	my_list+= list(np.repeat(L, M))
	my_list= tuple(my_list)
	value= np.random.standard_normal(size= my_list)
	return threshold, value

def get_accurate_partition(K, M, L, rho):
	threshold= [-rho/np.sqrt(math.pi), 0, rho/np.sqrt(math.pi)]
	threshold= np.tile(np.expand_dims(np.expand_dims(threshold, axis= 0), axis= 2),[K,1,M])
	print (threshold)
	my_list= [K]
	my_list+= list(np.repeat(L, M))
	my_list= tuple(my_list)
	value= np.random.standard_normal(size= my_list)
	return threshold, value


def get_return(T, N, M, rho, sigma_F_2, K, L, random):
	characteristics= get_characteristics(T, N , M, rho)
	# print (characteristics.shape)
	if random:
		threshold, value= get_accurate_partition(K, M ,L, rho)		
	else:
		threshold, value= get_partition(K, M ,L)
	loading= get_loading(characteristics, threshold, value)
	# print (loading[0,:,:].dot(loading[0,:,:].T))
	# print (loading[1,:,:].dot(loading[1,:,:].T))
	# print (loading.shape)
	factor= get_factor(T, K, sigma_F_2)
	print (np.mean(factor, axis= 0))
	factor_expand= np.expand_dims(factor, 1)
	# print (factor.shape)
	noise= get_idiosyncratic(T, N)
	return_data= np.zeros((T, N))
	for i in range(T):
		return_data[i,:]= np.squeeze(factor_expand[i,:,:].dot(loading[i,:,:]), 0)
	return_data= return_data+ noise
	return return_data, characteristics, factor

return_data, characteristics, factor= get_return(args.T, args.N, args.M, args.rho, args.sigma_F_2, args.K, args.L, args.random)
print (return_data.shape)
print (characteristics.shape)
path= './npz_data/simulated_data'
np.savez(path, return_data= return_data, characteristics= characteristics)
path= './npz_data/simulated_factor'
np.savez(path, factor= factor)


