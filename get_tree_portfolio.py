import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('--depth', default= 4, help='0 means the Weber data, 1 means Markus data, 2 means processed data of my own')
args = parser.parse_args()

def get_tree(depth, res, nodetag,  value_array, L, d, chara_data, oldlist, parent):
	L, N= chara_data.shape	
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
					# print (temp_array==value_array)
					get_tree(depth + 1, res, nodetag, temp_array, L, d, chara_data, newlist, parent)
				elif j == 1:
					temp_array[(index) & (chara_data[i, :] >= data_median), i] = 0
					get_tree(depth + 1, res, nodetag, temp_array, L, d, chara_data, newlist, parent)
				parent.pop()


def form_portfolio(returns, chara_data):
	T, N, L= chara_data.shape
	d= int(args.depth)
	portfolio = np.zeros((T, (2*L)**d))
	for i in range(T):
		if i%20==0:
			print ('Have gone through %d' %(i))
		res = []
		nodetag= []	
		get_tree(0, res, nodetag, np.ones([N, L]), L, d, chara_data[i,:].T, [], [])
		for j in range(len(res)):
			portfolio[i,j]= returns[i,:].dot(res[j])
		path= './npz_data/simulated_portfolio'
		np.savez(path, portfolio= portfolio)
data= np.load('./npz_data/simulated_data.npz')
returns= data['return_data']
characteristics= data['characteristics']
form_portfolio(returns, characteristics)
