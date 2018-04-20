import numpy as np
import pandas as pd

data = np.load('NpForMartinLettau.npz')

empirical_data = data['data']
date = data['date']

UNK = -99.99
empirical_shape = empirical_data.shape[1]

permno_array = np.arange(empirical_shape)
for i in range(len(date) - 360, len(date)):
    perm_chara = empirical_data[i, :]
    # permno_characteristic = perm_chara[:, [1, 8, 17, 33, 37]]
    permno_characteristic = perm_chara
    # An import thing is to see which characteristics
    #  corresponds to size and book to market.I do it manually.
    index = permno_characteristic == UNK
    tempindex = np.sum(index, axis=1, keepdims=True)
    useful_index = np.where(tempindex == 0)[0]
    permno_array = np.intersect1d(permno_array, useful_index)
    print(len(permno_array))

processed_data = empirical_data[len(date) - 360:len(date)]
processed_data = processed_data[:, permno_array]

path_target = 'C:/Users/zihan/Desktop/Files/Research/bryankelly_without_missing.npz'
print('Saving data to ' + path_target)
np.savez(path_target, processed_data=processed_data, stockdate= date[len(date)-360:len(date)])
print('Finished! ')
