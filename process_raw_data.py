import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('--tic', default= 'RET,AT,A2ME,BEME,Idio_vol,LME,r12.2', help='Please split by ,')
parser.add_argument('--start_time',default= 198401, help= 'Please type in the starting time')
parser.add_argument('--end_time', default= 201401, help= 'Please type in the ending time')
parser.add_argument('--total_time', default= 360, help= 'please type in the total time of the interval')
args = parser.parse_args()


File_name= args.tic.split(',')
print (File_name)
all_firm_name= []
mapping_relation= []
for file in File_name:
	file_msg = './Jason_data/%s.csv' % (file)
	# print(file_msg)
	file_data= pd.read_csv(file_msg)
	firm_name= file_data.axes[1][1:]
	# print (file_data.axes[1])
	dates= file_data['Date']
	dates= dates//100
	dates_list= list(dates)
	# print (dates_list)
	time_idx= {date:i for i, date in enumerate(dates_list)}
	# print (args.start_time)
	# print (time_idx.keys())
	idx_start = time_idx[args.start_time]
	idx_end= time_idx[args.end_time]
	possible_name= [i for i in range(len(firm_name))]
	# possible_name: the index of all of the firms.
	# print (idx_start)
	# print (idx_end)
	for i in range(idx_start, idx_end):
		data_i= file_data.iloc[i+1][1:]
		# print(possible_name)
		possible_name_copy= copy.deepcopy(possible_name)
		for j in possible_name:			
			# print (j)
			if np.isnan(data_i[j]):
				possible_name_copy.remove(j)
				# print (possible_name_copy)
		possible_name= possible_name_copy
	this_firm_name= list(firm_name[possible_name])
	
	for i in range(len(possible_name)):
		name= this_firm_name[i]
		this_firm_name[i]= name.replace('%s.'%(file), '')
		# print (name)
	this_mapping= {this_firm_name[i]: possible_name[i] for i in range(len(this_firm_name))}
	print (this_firm_name)
	all_firm_name.append(this_firm_name)
	mapping_relation.append(this_mapping)

# Get the intersection of all of the firm names.
intersection= []
for i in range(len(all_firm_name)):
	if i==0:
		intersection= all_firm_name[0]
	else:
		intersection= [item for item in intersection if item in all_firm_name[i]]

# Get the clean data, and save the clean data.
count= 0
data= np.zeros((args.total_time, len(intersection), len(File_name)))
for file in File_name:
	file_msg = './Jason_data/%s.csv' % (file)
	file_data= pd.read_csv(file_msg)
	# print (intersection[0])
	# print (mapping_relation[0].keys[0])
	# print (mapping_relation[0])
	# print (mapping_relation[0][intersection[0]])
	map_idx= []
	for i in range(len(intersection)):
		mapping_relation[count][intersection[i]]+= 1
		map_idx.append(mapping_relation[count][intersection[i]])
	# print (map_idx)
	time_idx= {date: i+1 for i, date in enumerate(dates_list)}
	idx_start = time_idx[args.start_time]
	idx_end= time_idx[args.end_time]
	for i ,idx in enumerate(range(idx_start, idx_end)):
		data[i, :, count]= file_data.iloc[idx][map_idx]
	count+= 1
path_target = 'C:/Users/zihanlin/Desktop/File/Research/npz_data/processed_data.npz'
np.savez(path_target, data= data)





				
	# idx_end= file_data.loc[dates== args.end_time]
	

