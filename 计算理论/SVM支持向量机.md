# 支持向量机(SVM)调研报告



### 姓名： 祝辰煜   学号：3023244369   班级：人工智能4班


## 1. 概述

### 1.1 什么是支持向量机？

支持向量机（Support Vector Machine，SVM）是一种监督学习模型，主要用于分类和回归分析。SVM通过构建一个或多个超平面（Hyperplane）在特征空间中分隔不同类别的数据点，以达到分类的目的。

### 1.2 支持向量机的历史背景和应用场景

SVM由Vladimir Vapnik和Alexey Chervonenkis在20世纪60年代末至90年代中期发展起来。最初，它被用于二分类问题，后来扩展到多分类问题。SVM在处理高维数据时表现出色，广泛应用于文本分类、人脸识别、手写数字识别、基因表达数据分析等领域。



## 2. 支持向量机的工作原理

### 2.1 线性SVM的基本概念

线性SVM旨在找到一个最佳的超平面，将数据集中的不同类别分开。这个最佳超平面是指到最近的数据点（支持向量）距离最大的超平面，如图1所示。

![svm](../image/svm.png)图1. 线性SVM的超平面及其分类示意图



### 2.2 非线性SVM及核技巧

在现实世界中，数据通常是线性不可分的。为了解决这个问题，SVM引入了核技巧（Kernel Trick）。通过将数据映射到更高维的空间中，使得数据在这个新空间中线性可分。常用的核函数包括：

- 多项式核函数（Polynomial Kernel）
- 高斯径向基函数（RBF Kernel）
- Sigmoid核函数



### 2.3 向量运算在SVM中的应用

向量运算是SVM的核心。特别是，支持向量机的分类规则是基于向量点积的。对于给定的输入数据点 \( x \)，我们通过计算其与支持向量的点积，并使用核函数，将其映射到更高维度的空间中。



### 2.4 SVM的优化问题

SVM的目标是最大化超平面的边界（Margin）这可以通过求解以下优化问题实现：

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
$$

在约束条件下：

$$
y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall i
$$

其中，$ \mathbf{w} $ 是超平面的法向量，$ b $ 是偏置，$ y_i $ 是数据点 $ \mathbf{x}_i $ 的类别标签。

## 3. 向量运算在SVM中的应用

### 3.1 向量点积在分类中的作用

在SVM中，决策函数可以表示为输入向量 $ \mathbf{x} $ 与支持向量 $ \mathbf{x}_i $ 的加权点积：

$$
f(\mathbf{x}) = \sum_{i=1}^N \alpha_i y_i (\mathbf{x}_i \cdot \mathbf{x}) + b
$$

其中，$ \alpha_i $ 是拉格朗日乘子，$ y_i $ 是类别标签。点积的计算是SVM进行分类决策的核心操作。

### 3.2 线性可分与不可分问题的处理

对于线性可分的问题，SVM直接通过超平面进行分类。而对于线性不可分的问题，SVM通过引入松弛变量（Slack Variables）来允许部分数据点位于错误的一侧，从而实现软间隔（Soft Margin）分类。

### 3.3 核函数及其常见类型

核函数通过计算输入向量在高维特征空间中的点积，帮助SVM处理非线性问题。常见的核函数包括：

- 线性核：$ K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j $
- 多项式核：$ K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + c)^d $
- 高斯RBF核：$ K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) $



## 4. Scikit-learn中的SVM实现



### 4.1 Scikit-learn中的SVM模块介绍

Scikit-learn是一个广泛使用的Python机器学习库，提供了便捷的接口来实现SVM。其 `sklearn.svm` 模块中包含了多种SVM变体，如 `SVC`（支持向量分类）、`SVR`（支持向量回归）、`LinearSVC` 等。



### 4.2 基于Scikit-learn的SVM实现实例

以下是一个使用Scikit-learn实现SVM分类的实例，我们将使用著名的鸢尾花数据集（Iris Dataset）进行分类任务。

#### 实例代码：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 仅使用前两个特征进行可视化
y = iris.target

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM模型训练
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 可视化决策边界
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(model, X_train, y_train)
```

#### 代码解析：

1. **加载数据集**：使用Scikit-learn加载鸢尾花数据集，并选取前两个特征进行分类。
2. **数据集划分**：将数据集划分为训练集和测试集。
3. **数据标准化**：对数据进行标准化，以提高模型的性能。
4. **训练SVM模型**：使用线性核函数的SVM对训练数据进行训练。
5. **预测与评估**：对测试集进行预测，并输出混淆矩阵和分类报告。
6. **可视化决策边界**：绘制SVM的决策边界，展示模型的分类效果。



### 4.3 实验结果与分析

运行上述代码，我们可以得到分类报告和混淆矩阵，进一步评估模型的性能。决策边界图显示了SVM在二维特征空间中的分类能力。

![15](../image/15.png)

![svm_decision_boundary](../image/svm_decision_boundary.png)
