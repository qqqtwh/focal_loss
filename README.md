# FocalLoss简介

one stage目标检测的准确率一般不如two stage算法，其中之一的原因就是样本的类别不平衡导致。一张图像上有成千上万的先验框，其中匹配到真实框的就是正样本，反之就是负样本。负样本过多主要带来两个问题：

- 负样本占总loss的大部分
- 样本多是容易分类的样本

这就导致模型训练的精确度下降。

因此FocalLoss的两个特性就是：

- 控制正负样本的权重
- 控制容易分类样本和难分类样本的权重

## 1.控制正负样本的权重

FocalLoss是在交叉熵损失的基础上演变而来，以二分类为例，二叉熵损失函数如下：
$$
CE(p,y)=\left\{\begin{array}.-log(p)\quad\quad ,y=1\\-log(1-p)\quad ,y=0\end{array}\right.
$$
其中p是预测结果，y是真是标签。

想要降低负样本的影响，就给其施加一个系数$\alpha$，其范围是0到1，此时的损失函数变为：
$$
CE(p,y)=\left\{\begin{array}.-log(p)*\alpha\quad\quad\quad\quad\quad ,y=1\\-log(1-p)*(1-\alpha)\quad ,y=0\end{array}\right.
$$

## 2.控制容易分类和难分类样本的权重

样本的预测和其真是类别结果越接近，其就越容易分类，在二分类样本中：

- 对于正样本：p越大，样本越容易分类；p越小，即(1-p)越小，样本越不容易分类
- 对于负样本：p越小，样本越容易分类；p越大，即(1-p)越大，样本越不容易分类

$(1-p)^\gamma$代入公式即可控制容易分类和难分类样本的权重，其中$\gamma$为调制系数：
$$
CE(p,y)=\left\{\begin{array}.-log(p)*(1-p)^\gamma\quad\quad ,y=1\\-log(1-p)*(p)^\gamma\quad ,y=0\end{array}\right.
$$
综合两个系数得到最后的FL函数公式为：
$$
FocalLoss(p,y)=\left\{\begin{array}.-log(p)*\alpha*(1-p)^\gamma\quad\quad ,y=1\\-log(1-p)*(1-\alpha)*(p)^\gamma\quad ,y=0\end{array}\right.
$$

## 3.代码

```python
import numpy as np
print('共有四个样本，它们依次为：\n正易分类样本，负易分类样本，正难分类样本，负难分类样本')
alpha = 0.25
gamma = 3
y = np.array([1,0,1,0])
p = np.array([0.95,0.05,0.5,0.5])
alpha_w = [alpha if yelem==1 else 1-alpha for yelem in y]
print('样本的alphw权重',alpha_w)

pt = np.zeros(4)
index1 = np.argwhere(y==1)
index0 = np.argwhere(y==0)

pt[index1] = (1-p[index1])**gamma
pt[index0] = (p[index0])**gamma
fl_w = pt*alpha_w
print('FocalLoss权重',fl_w)
'''
共有四个样本，它们依次为：
正易分类样本，负易分类样本，正难分类样本，负难分类样本
样本的alphw权重 [0.25, 0.75, 0.25, 0.75]
FocalLoss权重 [0.00003125 0.00009375 0.03125 0.09375]
'''
```

可以看到所有难分类样本的权重变大