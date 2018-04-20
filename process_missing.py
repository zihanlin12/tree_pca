import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import random

parser = argparse.ArgumentParser(description='Process Raw Data')
# parser.add_argument('--tic', default= 'AT,A2ME,BEME,IdioVol,LME,r12_2,LT_Rev,ST_Rev', help='Please split by ,')
parser.add_argument('--tic', default= 'BEME,LME,Idio_vol,AT,A2ME,r12.2', help='Please split by ,')
# ,IdioVol,AT,A2ME,r12_2
# parser.add_argument('--start_time',default= 196401, help= 'Please type in the starting time')
# parser.add_argument('--end_time', default= 201612, help= 'Please type in the ending time')
parser.add_argument('--start_time',default= 198401, help= 'Please type in the starting time')
parser.add_argument('--end_time', default= 201401, help= 'Please type in the ending time')
parser.add_argument('--total_time', default= 360, help= 'please type in the total time of the interval')
args = parser.parse_args()


File_name= args.tic.split(',')
File_name= list(File_name)
print (File_name)


# return_msg= './Jason_data/RET.csv'
return_msg= './Characteristics/RET.csv'
return_data= pd.read_csv(return_msg)
dates= return_data['date']
dates= dates//100
dates_list= list(dates)
firm_names= return_data.axes[1][1:]
firm_names= list(firm_names)
for i in range(len(firm_names)):
	name= firm_names[i]
	firm_names[i]= name.replace('RET.', '')
time_idx= {date:i for i, date in enumerate(dates_list)}
possible_name= [i for i in range(len(firm_names))]

idx_all_time= []
idx_start= time_idx[args.start_time]
idx_end= time_idx[args.end_time]
for i in range(idx_start+1, idx_end):
	data_i= return_data.iloc[i+1][1:]
	idx_name_i= [k for k in possible_name if not np.isnan(data_i[k])]
	idx_all_time.append(idx_name_i)

file_dic= {}
for file in File_name:
	# file_msg = './Jason_data/%s.csv' % (file)
	file_msg = './Characteristics/%s.csv' % (file)
	print (file_msg)
	file_data= pd.read_csv(file_msg, engine='python')
	# print (file_data)
	file_dic[file_msg]= file_data
	firm_name= file_data.axes[1][1:]
	dates= file_data['Date']
	dates= dates//100
	dates_list= list(dates)
	time_idx= {date:i for i, date in enumerate(dates_list)}

	firm_name_list= list(firm_name)
	for i in range(len(firm_name_list)):
		name= firm_name_list[i]
		firm_name_list[i]= name.replace('%s.'%(file), '')
	firm_idx= {firm:i for i, firm in enumerate(firm_name_list)}

	idx_start = time_idx[args.start_time]
	idx_end= time_idx[args.end_time]
	possible_name= [i for i in range(len(firm_name))]
	count= 0
	for i in range(idx_start, idx_end-1):
		data_i= file_data.iloc[i+1][1:]
		# print (len(idx_all_time[count]))
		possible_name_copy= copy.deepcopy(idx_all_time[count])
		for j in idx_all_time[count]:
			firm_name= firm_names[j]
			idx= firm_idx.get(firm_name,-999)
			# print (np.isnan(data_i[idx]))
			if np.isnan(data_i[idx]) or idx == -999:
				possible_name_copy.remove(j)
				# print ('I have been here')
		idx_all_time[count]= possible_name_copy
		# print (len(idx_all_time[count]))
		test_list= [firm_idx.get(firm_names[k], -999)+1 for k in idx_all_time[count]]
		# print (file_data.iloc[i+1][test_list])
		count+=1 

N= min([len(idx_all_time[count]) for count in range(len(idx_all_time))])
print (N)

data= np.zeros((idx_end-idx_start-1, N, len(File_name)+1))
count= 0
for i in range(idx_start, idx_end-1):
	# print (i)
	idx_i= random.sample(idx_all_time[count], N)
	idx_return= [k+1 for k in idx_i]
	# print (return_data.iloc[i+2][idx_return])
	data[count, :, 0]= return_data.iloc[i+2][idx_return]
	for j, file in enumerate(File_name):
		# file_msg= './Jason_data/%s.csv'%(file)
		file_msg= './Characteristics/%s.csv'%(file)
		# print (file_msg)
		# file_data= pd.read_csv(file_msg)
		file_data= file_dic[file_msg]
		firm_name= file_data.axes[1][1:]
		firm_name_list= list(firm_name)
		for k in range(len(firm_name_list)):
			name= firm_name_list[k]
			firm_name_list[k]= name.replace('%s.'%(file), '')
		firm_idx= {firm:n for n, firm in enumerate(firm_name_list)}
		firm_j= [firm_names[k] for k in idx_i]
		idx_file= [firm_idx.get(k, -999)+1 for k in firm_j]
		# print (file_data.iloc[i+1][idx_file])
		data[count, :, j+1]= file_data.iloc[i+1][idx_file]
	count+= 1
path_target = './npz_data/processed_data.npz'
np.savez(path_target, data= data)
# path_target = './npz_data/processed_data.csv'
# np.savetxt(path_target, data, delimiter=",") 
	

