# Plot the importance of the relative 

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import pandas as pd
# from matplotib.ticker import PercentFormatter
import argparse

parser= argparse.ArgumentParser(description= 'Plot Sharpe Ratio')
parser.add_argument('--file', default= './data/importance.csv', help= 'File path', type= str)
parser.add_argument('--ylabel', default= 'Which portfolios matter', help= 'In sample/out of sample Sharpe Ratio', type= str)
args= parser.parse_args()

def plot(sharpe_ratio, ticker, ylabel):
	N_type= sharpe_ratio.shape[0]
	print (N_type)
	width= 1.0/(N_type+1)
	# print (width)
	# print (width*N_type)
	fig, ax= plt.subplots()
	rects= dict()
	for i in range(N_type):
		c= (i+0.5)/N_type
		rects[i]= ax.bar(width*i, abs(sharpe_ratio[i]), width, color= (c,c,1-c))
	ax.set_ylabel(ylabel)
	ax.set_xticks(np.linspace(0, width*N_type, N_type))
	print (np.arange(width, width*N_type, N_type))
	ax.set_xticklabels(ticker, rotation = 15, ha="right")
	# ax.legend((rects[i] for i in range(N_type)), type_sharpe)
	plt.show()
	# plt.savefig('./data/OOSSharpeRatioShort.png')
	# plt.clf()

def plot_file(path, ylabel):
	df= pd.read_csv(path, header= None)
	df= pd.DataFrame.as_matrix(df)
	ticker= df[:,0]
	# print (ticker)
	sharpe_ratio= df[:,1]
	# print (sharpe_ratio)
	# header= df.axes[1]
	# # type_sharpe= list(header[1:])
	# ticker= list(df[header[0]])
	# print (ticker)
	# sharpe_ratio= df[header[1:]].values
	ticker, sharpe_ratio= sort_label(ticker, sharpe_ratio)
	plot(sharpe_ratio, ticker, ylabel)

def sort_label(ticker,sharpe_ratio):
	sharpe_ratio= np.abs(sharpe_ratio)
	index= np.argsort(sharpe_ratio)
	index= index[::-1]
	ticker= ticker[index]
	sharpe_ratio= sharpe_ratio[index]
	return ticker, sharpe_ratio

def beautiful_print(alist):
    char_list= ['momentum', 'size', 'Idio_vol', 'beme','Book']
    size_list= ['small', 'big']
    out_list= []
    for char in alist:
        out_list.append((char_list[char[0]], size_list[char[1]]))
    return out_list

if __name__ == '__main__':
	plot_file(args.file, args.ylabel)