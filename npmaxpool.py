import numpy as np

# numpy实现maxpool并返回权重的原理

x = np.random.normal(size=(2, 8, 3))
mx = np.max(x, axis=1, keepdims=True)

omx = np.where(x == mx, mx, 0)
print(x)
print(mx)
print(omx)
