

# Python矩阵运算

**姓名： 祝辰煜   学号：3023244369   班级：人工智能4班



### 运行效果

![01](D:\文件\同步空间\BaiduSyncdisk\学习资料\人工智能数学\作业\image\01.jpg)





### 源代码

```python
import numpy as np
from math import sqrt

# 1. 定义矩阵
A = np.array([[1, 2, 3], 
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

# 2. 矩阵加法
C = A + B
print("矩阵加法:")
print(C)
print()

# 3. 矩阵减法 
D = A - B
print("矩阵减法:")
print(D)
print()

# 4. 矩阵数乘
k = 3
E = k * A
print("矩阵数乘:")
print(E)
print()

# 5. 矩阵乘积
F = np.matmul(A, B)
print("矩阵乘积:")
print(F)
print()

# 6. 矩阵点乘
G = A * B
print("矩阵点乘:")
print(G)
print()

# 7. 矩阵转置
H = A.T
print("矩阵转置:")
print(H)
print()

# 8. 矩阵求逆
try:
    I = np.linalg.inv(A)
    print("矩阵求逆:")
    print(I)
except np.linalg.LinAlgError:
    print("矩阵A不可逆")
print()

# 9. 矩阵行列式
det_A = np.linalg.det(A)
print("矩阵A的行列式:")
print(det_A)
print()

# 10. 矩阵特征值与特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print("矩阵A的特征值:")
print(eigenvalues)
print("矩阵A的特征向量:")
print(eigenvectors)
```

