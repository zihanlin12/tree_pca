import numpy as np
import pandas as pd
import os
import argparse
# First of all, get the permutation of all of the characteristics.
parser = argparse.ArgumentParser(description='Process Raw Data')
parser.add_argument('--T', default= '360', help='Please split by ,')
args = parser.parse_args()


def permute(nums, depth):
	def _permute(result, temp, nums, k, depth):
		if k==depth:
			result.append(temp)
		else:
			for i in range(len(nums)):
				_permute(result, temp+[nums[i]],nums, k+1, depth)
	if nums is None:
		return []
	if len(nums)==0:
		return []
	result= []
	_permute(result, [], nums,0, depth)
	return result

# For this case, len(nums)= 3. Get all of the permutations first.
number_chara= 3
depth= 4
result= permute(range(1,1+number_chara), depth)
print (result)

T= int(args.T)

portfolio= np.zeros((T, ((2*number_chara)**depth)))

length= 2**depth

for i, item in enumerate(result):
	print (item)
	item= ''.join(str(i) for i in item)
	file_name= os.getcwd()+'/lme_beme_r12_2/'+str(item)+'train_ret.csv'
	data= pd.read_csv(file_name)
	portfolio[:,length*i:length*(i+1)]= data.iloc[:,length-1:2*length-1]

file_name= './npz_data/tree_without_missing.npz'
np.savez(file_name, portfolio= portfolio)
print ('Finish Saving Data')

