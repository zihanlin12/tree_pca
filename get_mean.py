import numpy as np

def get_characteristics(T, N, M, rho):
	base= np.random.randn(T, N, M)
	corre= np.diag(rho*np.ones((M-1)), 1)+ np.identity(M)
	for i in range(T):
		base[i,:,:]= base[i,:,:].dot(corre)
	return base


characteristics= get_characteristics(1, 10000000, 2, 0.7)
T= 1
for i in range(T):
	print(np.median([characteristics[i,k,1] for k in range(10000000) if characteristics[i,k,0]>0]))
