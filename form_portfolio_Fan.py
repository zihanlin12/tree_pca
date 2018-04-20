import argparse
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--J', default=2, help='How many to pick for each characteristics')
parser.add_argument('--M', default= 3, help= 'The order of characteristics')
parser.add_argument('--tic', default='Return,BMclean,Bookclean,MOMclean,vol', help= 'All of the characteristics')
args = parser.parse_args()

File_name= args.tic.split(',')
print (File_name)

J= args.J
M= args.M

# ret= File_name[0]
# chara= File_name[1:]
# T= 360 
# N= 720
# L= len(File_name)-1
# print (L)
# file_msg = './cleaned_data/%s.csv' % (ret)
# df = pd.read_csv(file_msg, header= None)
# returns = pd.DataFrame.as_matrix(df)
# chara_data= np.zeros((T, N, L))
# for i, file in enumerate(chara):
# 	print (file)
# 	file_msg = './cleaned_data/%s.csv' % (file)
# 	df = pd.read_csv(file_msg, header= None)
# 	data= pd.DataFrame.as_matrix(df)
# 	chara_data[:,:,i]= data

# data= np.load('./npz_data/Np_processed.npz')
# data= data['data']
# T, N, L= data.shape
# T= T+1
# returns= np.zeros((T, N))
# returns[1:]= data[:,:,0]
# index= [1]
# index+= list(range(5,L))
# chara_data= data[:,:,index]
# L= chara_data.shape[2]

data= np.load('./npz_data/processed_data.npz')
data= data['data']
T, N, L= data.shape
T= T+1
returns= np.zeros((T, N))
returns[1:]= data[:,:,0]
# index= [1]
# index+= list(range(5,L))
chara_data= data[:,:,1:]
L= chara_data.shape[2]
print (L)

fama_french_data= np.load('./npz_data/riskfreerate.npz')
fama_french_data= fama_french_data['fama_french_data']
r_f= fama_french_data[:,4]
fama_french_date= fama_french_data[:,0]
r_f= r_f/100
date_start= np.asscalar(np.argwhere(fama_french_date==198401))
date_end= np.asscalar(np.argwhere(fama_french_date==201401))
returns= returns-r_f[date_start:date_end:,np.newaxis]

portfolio= np.zeros((T-1, N))

for i in range(T-1):
	chara_data_i= chara_data[i,:,:]
	basis= np.ones((N, (J+M-1)*L+1))
	print (basis.shape)
	for j in range(L):
		sample= chara_data_i[:,j]
		# print (sample.shape)
		# Dimension of sample: p
		quantile= np.array([np.percentile(sample, int(100.0/(J+1)*(k))) for k in range(1,J+1)])
		# print (quantile.shape)
		# Dimension of sample: J
		basis[:,j*(J+M-1):(j*(J+M-1)+J)]= np.maximum(np.tile(sample[:,np.newaxis], J)-np.tile(quantile[:, np.newaxis], N).T, 0)
		basis[:,j*(J+M-1):(j*(J+M-1)+J)]= np.power(basis[:,j*(J+M-1):(j*(J+M-1)+J)] , M-1)
		basis[:,(j*(J+M-1)+J):(j*(J+M-1)+J+M-1)]=  np.stack([np.power(sample, k) for k in range(1, M)], axis= 1)
	basis[:,-1]= np.power(sample,0)
	# print (np.linalg.matrix_rank(basis))
	# print ((basis.T.dot(basis)).shape)
	portfolio[i,:]= returns[i+1, :].dot(basis).dot(np.linalg.inv(basis.T.dot(basis))).dot(basis.T)
	# portfolio[i,:]= returns[i+1, :].dot(basis).dot(np.linalg.inv(basis.T.dot(basis)))

path= './npz_data/Fan_portfolio'
np.savez(path, portfolio= portfolio)


# Form the 