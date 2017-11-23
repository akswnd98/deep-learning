import numpy as np
import matplotlib.pylab as plt
import pickle
from PIL import Image
from exam2 import *

f = open('params.dat', 'rb')
network = pickle.load(f)
f.close()

f = open('t10k-images.idx3-ubyte', 'rb')
f.seek(16, 0)
x = np.empty((10000, 784), dtype='float64')
for i in range(10000):
    x[i] = np.float64(np.frombuffer(f.read(784), dtype='uint8')) / 256
f.close()

f = open('t10k-labels.idx1-ubyte', 'rb')
f.seek(8, 0)
t = np.empty(10000, dtype='int')
for i in range(10000):
    t[i] = np.int(np.frombuffer(f.read(1), dtype='uint8'))
f.close()

