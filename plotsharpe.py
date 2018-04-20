import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser= argparse.ArgumentParser(description= 'Plot Sharpe Ratio')
parser.add_argument('--file', default= './data/tree_individual.csv', help= 'File path', type= str)
parser.add_argument('--ylabel', default= 'Out of sample Sharpe Ratio', help= 'In sample/out of sample Sharpe Ratio', type= str)
args= parser.parse_args()

def plot(sharpe_ratio, ticker, type_sharpe, ylabel):
	N_ticker, N_type= sharpe_ratio.shape
	ind=np.arange(N_ticker)
	width= 1.0/(N_type+1)
	fig, ax= plt.subplots()
	rects= dict()
	for i in range(N_type):
		c= (i+0.5)/N_type
		rects[i]= ax.bar(ind+width*i, sharpe_ratio[:,i], width, color= (c,c,1-c))
	ax.set_ylabel(ylabel)
	ax.set_xticks(ind+width*(N_type-1)/2)
	ax.set_xticklabels(ticker)
	ax.legend((rects[i] for i in range(N_type)), type_sharpe)
	# plt.show()
	plt.savefig('./data/tree_individual.png')

def plot_file(path, ylabel):
	df= pd.read_csv(path)
	header= df.axes[1]
	type_sharpe= list(header[1:])
	ticker= list(df[header[0]])
	sharpe_ratio= df[header[1:]].values
	plot(sharpe_ratio, ticker, type_sharpe, ylabel)

if __name__ == '__main__':
	plot_file(args.file, args.ylabel)




