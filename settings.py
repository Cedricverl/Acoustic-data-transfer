import numpy as np
sync = np.array([1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]*64, np.float32)
N = 482
Nd = N//2-1
L = 200
Nr = 200
Lt = 2
Ld = 8
b = 4
M = 2**b