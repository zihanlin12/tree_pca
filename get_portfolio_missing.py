# This code deals with missing data and form portfolios at the same time

import numpy as np
import pandas as pd
import copy
import argparse
import random


arser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('--tic', default= 'BEME,LME,Idio_vol,AT,A2ME,r12.2', help='Please split by ,')
parser.add_argument('--start_time',default= 198401, help= 'Please type in the starting time')
parser.add_argument('--end_time', default= 201401, help= 'Please type in the ending time')
args = parser.parse_args()

File_name= args.tic.split(',')
File_name= list(File_name)
print (File_name)

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