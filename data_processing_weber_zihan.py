import os
import numpy as np
import random 
import copy

def deco_print(line):
	print('>==================> ' + line)

current_dir= os.getcwd()
data_dir= current_dir+'/npz_data/NpForMartinLettau.npz'

data= np.load(data_dir)
Martin_data= data['data']
date= data['date']
Martin_data= Martin_data[-360:]
print (date[-360])
print (date[-1])

T,N,L= Martin_data.shape
print (T)
UNK = -99.99

M= 1000000
all_index= []
for i in range(1,T):
	returns= Martin_data[i,:, 0]
	chara_data= Martin_data[i-1,:,1:]
	index= [i for i in range(len(returns)) if returns[i]!=UNK]
	for k in range(chara_data.shape[1]):
		index_copy= copy.deepcopy(index)
		index= [i for i in index_copy if chara_data[i,k]!= UNK]		
	all_index.append(index)
	M= min(M, len(index))
print (M)

data_output= np.zeros((T-1, M, L))
for i in range(1,T):
	returns= Martin_data[i,:,0]
	chara_data= Martin_data[i-1,:,1:]
	index= random.sample(all_index[i-1], M)
	data_output[i-1,:,0]= returns[index]
	data_output[i-1,:, 1:]= chara_data[index,:]


path_target= current_dir+'/npz_data/Np_processed.npz'
deco_print('Saving data to ' + path_target)
np.savez(path_target, data=data_output)
deco_print('Finished!')

