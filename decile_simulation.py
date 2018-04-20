import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--D', default= 10, help= 'The number of stocks in each portfolio')
args = parser.parse_args()

def form_portfolio(returns, chara_data):
	T, N, L= chara_data.shape
	D= int(args.D)
	portfolio= np.zeros((T, L*D))
	for i in range(T):
		for j in range(L):
			chara_data_t= chara_data[i,:,j]
			index= np.argsort(chara_data_t)
			length= int(len(index)/D)
			for k in range(D):
				index_k= index[int(length*k):int(length*(k+1))]
				portfolio[i, j*D+k]= np.mean(returns[i,index_k])
	path= './npz_data/simulated_decile_portfolio'
	np.savez(path, portfolio= portfolio)
data= np.load('./npz_data/simulated_data.npz')
returns= data['return_data']
characteristics= data['characteristics']
form_portfolio(returns, characteristics)