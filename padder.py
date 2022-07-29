import time

import numpy as np
from operator import xor

N = 3
K = 9
import time


def my_pad(vector):
    return np.bitwise_xor(bool(vector[N]), bool(vector[K]))


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = np.bitwise_xor(bool(vector[N]), bool(vector[K]))
    print('padv ', pad_value)
    # vector[pad_width[0]] = pad_value
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def paddd_with(vector, pad_width, iaxis, kwargs):
    pad_value = np.bitwise_xor(bool(vector[N]), bool(vector[K]))
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


a = np.ones(3)
a = np.append(a, 4)
print(a)

a1 = np.roll(a, 3)
print(a1)

a2 = np.pad(a, (0, 0), mode='constant')

print(a2)

print('________\n')
a = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
print(a, len(a))

# a = np.pad(a, 1, pad_with)[:-1]
k = 5
a = np.pad(a, k, pad_with)[:-int(k*2)]
g = []
g.append(np.pad(a, k, pad_with)[:-int(k*2)])
print(a, len(a))
