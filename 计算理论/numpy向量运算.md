# numpy 向量运算

### 姓名： 祝辰煜   学号：3023244369   班级：人工智能4班
### 实验代码

```python
import numpy as np
import math

# 定义向量
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 向量加法
v_add = v1 + v2
print("向量加法:", v_add)

# 向量减法
v_sub = v1 - v2
print("向量减法:", v_sub)

# 向量数乘
scalar = 3
v_scalar = scalar * v1
print("向量数乘:", v_scalar)

# 向量点乘
dot_product = np.dot(v1, v2)
print("向量点乘:", dot_product)

# 向量模长(范数)
v1_norm = np.linalg.norm(v1)
v2_norm = np.linalg.norm(v2)
print("向量 v1 的模长:", v1_norm)
print("向量 v2 的模长:", v2_norm)

# 向量夹角
angle = math.acos(dot_product / (v1_norm * v2_norm))
print("向量夹角(弧度):", angle)
print("向量夹角(度):", math.degrees(angle))
```



### 运行效果截图

![10](../image/10.png)
