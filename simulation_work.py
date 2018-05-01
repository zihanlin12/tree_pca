import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('--T', default=638, help='Whether use Risk Premium PCA', type= int)
parser.add_argument('--N', default= 1000, help = 'Whether use correlation matrix', type= int)
parser.add_argument('--M', default= 2, help = 'Whether use correlation matrix', type= int)
parser.add_argument('--rho', default= 0.7, help = 'Whether use correlation matrix', type= float)
parser.add_argument('--sigma_F_2', default= 0.03, help = 'Whether use correlation matrix', type= float)
parser.add_argument('--K', default= 4, help = 'Whether use correlation matrix', type= int)
parser.add_argument('--L', default= 3, help = 'The number of partitions in each coordinate', type= int)
parser.add_argument('--random', default= 1, help= 'Whether we are simulating the ', type= str)
args = parser.parse_args()

K= int(args.K)

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
	diag= np.array([5, 0.3, 0.1, sigma_F_2])
	diag= diag[0:K]
	diag_F= np.sqrt(np.diag(diag))
	Factor= np.random.randn(T, K).dot(diag_F)
	Sharpe= np.array([5,0.1,0.3,0.5])
	Sharpe= Sharpe[0:K]
	print (np.mean(Factor, axis= 0))
	Factor= Factor+ Sharpe.dot(diag_F)
	return Factor

def get_characteristics(T, N, M, rho):
	base= np.random.randn(T, N, M)
	corre= np.diag(rho*np.ones((M-1)), 1)+ np.identity(M)
	for i in range(T):
		base[i,:,:]= base[i,:,:].dot(corre)
	return base

def get_individual_loading(x, rho, B):
	M= len(x)
	output= np.zeros((K))
	basis= return_basis_matrix(x, rho)
	for i in range(K):		
		output[i]= np.sum(basis*B)
	return output

def get_coefficient():
	return np.random.rand(10)

def return_basis_matrix(x, rho):
	# coef= rho/np.sqrt(2*math.pi)
	coef= 0.547710011198
	basis_matrix= np.zeros(10)
	basis_matrix[0]= 1 if x[0]<0 and x[1]>0 else 0
	basis_matrix[1]= 1 if x[0]>-coef and x[0]<0 and x[1]<-coef else 0
	basis_matrix[2]= 1 if x[0]<-coef and x[1]>-coef and x[1]<0 else 0
	basis_matrix[3]= 1 if x[0]>-coef and x[0]<0 and x[1]>-coef and x[1]<0 else 0
	basis_matrix[4]= 1 if x[0]<-coef and x[1]<-coef else 0
	basis_matrix[5]= 1 if x[0]>0 and x[1]<0 else 0
	basis_matrix[6]= 1 if x[0]<coef and x[0]>0 and x[1]>coef else 0
	basis_matrix[7]= 1 if x[0]>coef and x[1]<coef and x[1]>0 else 0
	basis_matrix[8]= 1 if x[0]<coef and x[0]>0 and x[1]<coef and x[1]>0 else 0
	basis_matrix[9]= 1 if x[0]>coef and x[1]>coef else 0
	return basis_matrix

def get_loading(characteristics, rho):
	T, N, M= characteristics.shape
	output= np.zeros((T, K, N))
	B= get_coefficient()
	for i in range(T):
		for j in range(N):
			output[i,:,j]= get_individual_loading(characteristics[i,j,:], rho, B)
		# q, _= np.linalg.qr(output[i,:,:].T, mode= 'reduced')
		# output[i,:,:]= q.T* np.sqrt(N)
	return output

def get_return(T, N, M, rho, sigma_F_2, K, L, random):
	characteristics= get_characteristics(T, N , M, rho)
	loading= get_loading(characteristics, rho)
	factor= get_factor(T, K, sigma_F_2)
	print (np.mean(factor, axis= 0))
	factor_expand= np.expand_dims(factor, 1)
	noise= get_idiosyncratic(T, N)
	return_data= np.zeros((T, N))
	for i in range(T):
		return_data[i,:]= np.squeeze(factor_expand[i,:,:].dot(loading[i,:,:]), 0)
	# If I consider the case when there is no noise:
	# noise= 0
	return_data= return_data+ noise
	return return_data, characteristics, factor

return_data, characteristics, factor= get_return(int(args.T), int(args.N), int(args.M), args.rho, args.sigma_F_2, args.K, args.L, args.random)
print (return_data.shape)
print (characteristics.shape)
path= './npz_data/simulated_data'
np.savez(path, return_data= return_data, characteristics= characteristics)
path= './npz_data/simulated_factor'
np.savez(path, factor= factor)
