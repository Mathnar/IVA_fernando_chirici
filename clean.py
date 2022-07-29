import time

import numpy as np
import numba


x = np.array([[1, 2, 3], [4, 5, 6]])

a = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
N = 3
K = 5


#   print(np.pad(x,((0,0),(1,0)), mode='constant')[:, :-1])


def my_pad(vector):
    #print(np.bitwise_xor(bool(vector[N]), bool(vector[K])))
    return np.bitwise_xor(bool(vector[N]), bool(vector[K]))


pad_arr = np.pad(a, (1, 0), 'constant', constant_values=my_pad(a))[:-1]


@numba.jit
def roll_and_cycle(vect, iter):
    pad_arr = vect
    for i in range(0, iter):
        pad_arr = np.roll(pad_arr, 1)
        np.insert(pad_arr, 0, np.bitwise_xor(bool(pad_arr[N]), bool(pad_arr[K])))
        np.delete(pad_arr, -1)
        print('_', pad_arr)


#@numba.jit
def roll_and_cycle_2(vect, iter):
    pad_arr = vect
    for i in range(0, iter):
        pad_arr = np.pad(pad_arr, (1, 0), 'constant', constant_values=my_pad(pad_arr))[:-1]
        #print('_',pad_arr)


print('\n____________')


print('\n_', pad_arr)
t1 = time.time()
roll_and_cycle(a, 300000)
t2 = time.time()
tot = t2-t1
print('\n2',tot)

# 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0 -> VW: 1
# 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0 -> VW: 1
# 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0 -> VW: 0
# 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1

test = np.zeros(int(1e8))

t1 = time.time()
#roll_and_cycle(test, len(test))
t2 = time.time()
tot = t2-t1
print(tot)
