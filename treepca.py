import numpy as np

data = np.load('C:\\Users\\zihan\\Desktop\\Files\\Research\\bryankelly_without_missing.npz')
processed_data = data['processed_data']
T, N, M = processed_data.shape
returns = processed_data[:, :, 1]
fama_french_data = np.load('C:\\Users\\zihan\\Desktop\\Files\\Research\\riskfreerate.npz')
fama_french_data = fama_french_data['fama_french_data']
r_f = fama_french_data[:, 4]
r_f = r_f / 100
date = fama_french_data[:, 0]
date_start = np.asscalar(np.argwhere(date == 198406))
date_end = np.asscalar(np.argwhere(date == 201405))
returns = returns - (r_f[date_start:date_end + 1:, np.newaxis])
# The dimension of returns: T*N
# chara_data = processed_data[:, :, [8, 17, 20, 33, 37]]
# chara_data = processed_data[:, :, [8, 17, 20, 33, 37]]
# chara_data= processed_data[:,:,[8,17, 18 ,20,33, 37]]
# chara_data = processed_data[:, :, [8, 17, 33, 37]]
chara_data= processed_data[:,:,5:10]


def get_tree(depth, res, value_array, L, d, chara_data, parent):
    if depth == d:
        final_array = np.ones([N, 1])
        for k in range(L):
            temp = value_array[:, k].reshape([N, 1])
            final_array = final_array * temp
        # print(np.sum(final_array))
        if np.sum(final_array) < 10 ** (-3):
            print(final_array)
        final_array = 1.0 * final_array / np.sum(final_array)

        # if np.sum(final_array)<10**(-3):
        #     print (value_array)
        res.append([final_array])
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
                parent.append(i)
                if j == 0:
                    temp_array[(index) & (chara_data[i, :] <= data_median), i] = 0
                    debug_array= (chara_data[i, :] <= data_median)
                    get_tree(depth + 1, res, temp_array, L, d, chara_data, parent)
                elif j == 1:
                    temp_array[(index) & (chara_data[i, :] >= data_median), i] = 0
                    get_tree(depth + 1, res, temp_array, L, d, chara_data, parent)
                parent.pop()


L = chara_data.shape[2]
d = 4
res = []
# The first parameter is number of characteristics, and the second is the depth of the tree.
get_tree(0, res, np.ones([N, L]), L, d, chara_data[31, :].T, [])
