import numpy as np
fs = 20000
sync = np.array([1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]*64, np.float32)
N = 128*2
Nd = N//2-1
L = 200
Nr = 200
Lt = 2
Ld = 8
b = 2
M = 2**b
CHUNK = sync.size*4
CHANNELS = 1
threshold = 30

