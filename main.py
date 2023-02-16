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
print(fl_w==[0.00003125, 0.00009375 ,0.03125 ,0.09375])