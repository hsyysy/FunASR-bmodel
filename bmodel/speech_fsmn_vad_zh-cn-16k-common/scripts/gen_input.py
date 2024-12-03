import numpy as np

input1 = np.random.randn(1,6000,400).astype(np.float32)
input2 = np.random.randn(1,128,19,1).astype(np.float32)
input3 = np.random.randn(1,128,19,1).astype(np.float32)
input4 = np.random.randn(1,128,19,1).astype(np.float32)
input5 = np.random.randn(1,128,19,1).astype(np.float32)
np.savez("input_0.npz", speech=input1, in_cache0=input2, in_cache1=input3, in_cache2=input4, in_cache3=input5)
