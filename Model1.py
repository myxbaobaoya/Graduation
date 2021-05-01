import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import math

#
# train_number = 200
# train_temp=[]
# with open("train_data.txt") as tdata:
#     while True:
#         line=tdata.readline()
#         if not line:
#             break
#         train_temp.append([float(i) for i in line.split()])
# train_temp = np.array(train_temp)
#
# result_temp=[]
# with open("result_data.txt") as rdata:
#     while True:
#         line=rdata.readline()
#         if not line:
#             break
#         result_temp.append([float(i) for i in line.split()])
# result_temp = np.array(result_temp)
#
# x = np.mat(train_temp)
# x = torch.tensor(x).float()
# y = np.mat(result_temp)
# y = torch.tensor(y).float()
# z = np.mat('0.060 0.300 0.100 6.000')
# z = torch.tensor(z).float()


class NonLinearRegression(nn.Module):
    def __init__(self):
        super(NonLinearRegression, self).__init__()  # 初始化父类
        self.fc1 = nn.Linear(4, 10)  # 输入层,全连接层
        self.tanh = nn.Tanh()  # 激活函数
        self.fc2 = nn.Linear(10, 1)  # 一个隐藏层

    def forward(self, x):
        x = self.fc1(x)  # 全连接层
        x = self.tanh(x)  # 激活选择
        x = self.fc2(x)  # 隐藏层计算输出最终结果
        return x


# model = NonLinearRegression()
# criterion = nn.MSELoss()  # 损失函数为均方差
#
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# for i in range(len(list(model.parameters()))):  # 查看模型参数
#     print('params: %d' % (i + 1), list(model.parameters())[i].size())
#
#
# epochs = 1000
# for epoch in range(epochs):
#     epoch += 1
#     # 清空梯度参数
#     optimizer.zero_grad()
#     # 获得输出
#     outputs = model(x)
#     # 计算损失
#
#     loss = criterion(outputs, y)
#     # 反向传播
#     loss.backward()
#     # 更新参数
#     optimizer.step()


# z = np.mat('0.060 0.300 0.100 6.000')
# z = torch.tensor(z).float()
# print(model(z).data.numpy())
# variance = 0
# for i in range(train_temp.shape[0]):
#     predict = np.mat(train_temp[i,:])
#     predict = torch.tensor(predict).float()
#     predict = model(predict).data.numpy()
#     predict = predict[0,:][0]
#     train = result_temp[i,:][0]
#     variance = math.pow((predict-train),2) + variance
# variance = variance/train_number
# print(variance)
