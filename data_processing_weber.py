import time
import pandas as pd
import numpy as np

def deco_print(line):
	print('>==================> ' + line)

path= 'C:/Users/zihanlin/Desktop/File/Research/NpForMartinLettau.csv'
# path = '../../datasets/Weber/NpForMartinLettau.csv'
deco_print('Loading data from ' + path)
df = pd.read_csv(path)
deco_print('Loading finished! ')
header = df.axes[1]
date = sorted(list(df.date.unique()))
date2idx = {date_i:i for i, date_i in enumerate(date)}
permno = sorted(list(df.permno.unique()))
variable = list(header[4:])

N_permno = len(permno)
N_date = len(date)
N_variable = len(variable)

deco_print('Processing data')
start = time.time()
UNK = -99.99
data = np.full(shape=[N_date, N_permno, N_variable], fill_value=UNK, dtype=np.float64)
for i in range(N_permno):
	permno_i = permno[i]
	df_i = df.loc[df['permno'] == permno_i]
	for j in range(len(df_i['date'])):
		date_j = df_i['date'].iloc[j]
		date_j_idx = date2idx[date_j]
		data[date_j_idx,i,:] = np.array(df_i[header[4:]].iloc[j])
	if i % 100 == 1:
		elapsed_time = time.time() - start
		print('Elapsed / Estimated: %0.2fs / %0.2fs' %(elapsed_time, elapsed_time / i * N_permno), end ='\r')
deco_print('Data processing finished! ')

path_target = 'C:/Users/zihanlin/Desktop/File/Research/NpForMartinLettau.npz'
deco_print('Saving data to ' + path_target)
np.savez(path_target, date=date, variable=variable, permno=permno, data=data)
deco_print('Finished!')