import pandas as pd
import numpy as np
import os

current_dir= os.getcwd()
file_dir= current_dir+'/npz_data/processed_data.npz'
processed_data= np.load(file_dir)
data= processed_data['data']
returns= data[:,:,0]
chara_data= data[:,:,1:]
rf_dir= current_dir+ '/npz_data/riskfreerate.npz'
fama_french_data= np.load(rf_dir)
fama_french_data= fama_french_data['fama_french_data']
r_f= fama_french_data[:,4]
fama_french_date= fama_french_data[:,0]
r_f= r_f/100
date_start= np.asscalar(np.argwhere(fama_french_date==198402))
date_end= np.asscalar(np.argwhere(fama_french_date==201401))
returns= returns- r_f[date_start:date_end:,np.newaxis]

T, N= returns.shape
def get_tree(depth, res, nodetag,  value_array, L, d, chara_data, oldlist, parent):
    if depth == d:
        final_array = np.ones([N, 1])
        for k in range(L):
            temp = value_array[:, k].reshape([N, 1])
            final_array = final_array * temp
        if np.sum(final_array) < 10 ** (-3):
            print('An error over here')
        final_array = 1.0 * final_array / np.sum(final_array)
        res.append([final_array]) 
        nodetag.append(oldlist)
    else:
        for i in range(L):
            for j in range(2):
                if len(parent) != 0:
                    index = np.ones((N), dtype=bool)
                    for k in parent:
                        index = (index) & ((value_array[:, k] != 0))
                    data_median = np.median(chara_data[i, index])
                else:
                    index = (value_array[:, i] != 0)
                    data_median = np.median(chara_data[i, index])
                temp_array = value_array.copy()
                newlist= list(oldlist)
                newlist+=[(i,j)]
                parent.append(i)
                if j == 0:
                    temp_array[(index) & (chara_data[i, :] <= data_median), i] = 0
                    get_tree(depth + 1, res, nodetag, temp_array, L, d, chara_data, newlist, parent)
                elif j == 1:
                    temp_array[(index) & (chara_data[i, :] >= data_median), i] = 0
                    get_tree(depth + 1, res, nodetag, temp_array, L, d, chara_data, newlist, parent)
                parent.pop()
L= chara_data.shape[2]
d= 4
portfolio = np.zeros((T - 1, (2*L)**d))
for i in range(T-1):
    if i%20==0:
            print ('Have gone through %d' %(i))
    res = []
    nodetag= []
    get_tree(0, res, nodetag, np.ones([N, L]), L, d, chara_data[i,:].T, [], [])
    for j in range(len(res)):
        portfolio[i][j]= returns[i,:].dot(res[j])

path_target = current_dir+'/npz_data/tree_portfolio.npz'
print('Saving data to ' + path_target)
np.savez(path_target, portfolio=portfolio)
print('Finished!')