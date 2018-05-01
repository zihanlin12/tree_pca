# From three kinds of portfolios for comparison. Fama-French, Decile, and Fan

# First of all, form the Fan portfolio.

import argparse
import numpy as np
import pandas as pd
import copy
import random

parser = argparse.ArgumentParser(description='Process Raw Data')
# parser.add_argument('--tic', default= 'AT,A2ME,BEME,IdioVol,LME,r12_2,LT_Rev,ST_Rev', help='Please split by ,')
parser.add_argument('--tic', default= 'BEME,LME,r12.2', help='Please split by ,')
parser.add_argument('--start_time',default= 198701, help= 'Please type in the starting time')
parser.add_argument('--end_time', default= 201612, help= 'Please type in the ending time')
parser.add_argument('--total_time', default= 360, help= 'please type in the total time of the interval')
parser.add_argument('--J', default=2, help='How many to pick for each characteristics')
parser.add_argument('--M', default= 3, help= 'The order of characteristics')
args = parser.parse_args()

File_name= args.tic.split(',')
File_name= list(File_name)

file_name_dic= {}

fama_french_data= np.load('./npz_data/riskfreerate.npz')
fama_french_data= fama_french_data['fama_french_data']
r_f= fama_french_data[:,4]
fama_french_date= fama_french_data[:,0]
r_f= r_f/100
date_start= np.asscalar(np.argwhere(fama_french_date==198701))
date_end= np.asscalar(np.argwhere(fama_french_date==201612))
r_f= r_f[date_start:date_end]

return_msg= './Characteristics/RET.csv'
return_data= pd.read_csv(return_msg)
dates= return_data['date']
dates= dates//100
dates_list= list(dates)
firm_names= return_data.axes[1][1:]
possible_name= [i for i in range(len(firm_names))]
firm_names_Ret= list(firm_names)
for i in range(len(firm_names)):
	name= firm_names[i]
	firm_names_Ret[i]= name.replace('RET.', '')
file_name_dic['RET']= firm_names_Ret

file_dic= {}
for file in File_name:
	file_msg = './Characteristics/%s.csv' % (file)
	file_data= pd.read_csv(file_msg, engine='python')
	file_dic[file]= file_data
	firm_name= file_data.axes[1][1:]
	firm_name_list= list(firm_name)
	for i in range(len(firm_name_list)):
		name= firm_name_list[i]
		firm_name_list[i]= name.replace('%s.'%(file), '')
	firm_idx= {firm:i for i, firm in enumerate(firm_name_list)}
	file_name_dic[file]= firm_idx

time_idx= {date:i for i, date in enumerate(dates_list)}
idx_start= time_idx[args.start_time]
idx_end= time_idx[args.end_time]

D= 10
J= args.J
M= args.M
L= len(File_name)
total_time= int(args.total_time)
portfolio_decile= np.zeros((total_time, D*len(File_name)))
portfolio_Fan= np.zeros((total_time,(J+M-1)*L+1))

for t, i in enumerate(range(idx_start, idx_end+1)):
	print (t)
	return_data_i= return_data.iloc[i][1:]
	idx_name_i= [k for k in possible_name if not np.isnan(return_data_i[k])]
	firm_name_Ret_i=  [firm_names_Ret[k] for k in idx_name_i]

	for j, file in enumerate(File_name):
		firm_name_list= file_name_dic[file]
		file_data= file_dic[file]
		data_i= file_data.iloc[i][1:]
		idx_name_copy= copy.deepcopy(idx_name_i)
		chara_list=[]
		for item in idx_name_i:
			name= firm_names_Ret[item]
			idx= firm_name_list.get(name, -999)
			if idx==-999 or np.isnan(data_i[idx]):
				idx_name_copy.remove(item)
			else:
				chara_list.append(idx)
		chara_data_t= data_i[chara_list]
		returns= return_data_i[idx_name_copy]
		# Form the decile portfolio
		index= np.argsort(chara_data_t)
		length= int(len(index)/D)
		for k in range(D):
			index_k= index[int(length*k):int(length*(k+1))]
			portfolio_decile[t, j*D+k]= np.mean(returns[index_k])

		# Form the Fan portfolio
		# print (J)
		N=  len(chara_list)
		# print (len(chara_list))
		# print (len(idx_name_copy))
		basis= np.ones((N, (J+M-1)))
		sample= chara_data_t
		quantile= np.array([np.percentile(sample, int(100.0/(J+1)*(k))) for k in range(1,J+1)])
		basis[:,0:J]= np.maximum(np.tile(sample[:,np.newaxis], J)-np.tile(quantile[:, np.newaxis], N).T, 0)
		basis[:,0:J]= np.power(basis[:,0:J] , M-1)
		basis[:,J:J+M-1]=  np.stack([np.power(sample, k) for k in range(1, M)], axis= 1)
		portfolio_Fan[t,j*(J+M-1):(j+1)*(J+M-1)]= returns.dot(basis)
	portfolio_Fan[t,-1]= np.mean(returns)

		# Form the Fama-French for 3 characteristics.
		# if len(File_name)<=3:c	
path_target = './npz_data/portfolio_without_missing_decile.npz'
np.savez(path_target, portfolio= portfolio_decile)
path_target = './npz_data/portfolio_without_missing_Fan.npz'
np.savez(path_target, portfolio= portfolio_Fan)









	

