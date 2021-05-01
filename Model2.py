import torch
import torch.nn as nn
import numpy as np
import math

import matplotlib.pyplot as plt

# train_number=200
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
# z = np.mat('[0.060 0.300 0.100 6.000]')
# z = torch.tensor(z).float()


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()  # 用nn.Module的init方法
        self.linear1 = nn.Linear(input_dim,10)  # 因为我们假设的函数是线性函数
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10,output_dim)

    def forward(self, x):
        # out = self.linear(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# input_dim = 4
# output_dim = 1
# model = LinearRegressionModel(input_dim, output_dim)
# criterion = nn.MSELoss()  # 损失函数为均方差
#
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# x = np.mat(train_temp)
# x = torch.tensor(x).float()
# y = np.mat(result_temp)
# y = torch.tensor(y).float()
# '''训练网络'''
# epochs = 1000
# for epoch in range(epochs):
#     epoch += 1
#     # 清空梯度参数
#     optimizer.zero_grad()
#     # 获得输出
#     outputs = model(x)
#     # 计算损失
#     loss = criterion(outputs, y)
#     # 反向传播
#     loss.backward()
#     # 更新参数
#     optimizer.step()


#print(y)
# predicted = model(z).data.numpy()
# print(predicted)
# plt.plot(train_temp,result_temp,'ro',label = 'train data')
# plt.show()
#torch.save(model, 'model.pkl')
#model = torch.load('model.pkl')

# variance = 0
# for i in range(train_temp.shape[0]):
#     predict = np.mat(train_temp[i,:])
#     predict = torch.tensor(predict).float()
#     predict = model(predict).data.numpy()
#     predict = predict[0,:][0]
#     train = result_temp[i,:][0]
#     variance = math.pow((predict-train),2) + variance
# variance = variance/train_number
# #print(variance)