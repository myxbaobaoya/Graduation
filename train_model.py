import torch.nn as  nn
import torch
import numpy as np
import math
import sys
from sys import path
import Model1
import Model2
import Model3
from matplotlib import pyplot as plt
import os


class train():
    def __init__(self):
        self.train_number = 200
        self.model11 =  Model1.NonLinearRegression()
        self.model21 =  Model2.LinearRegressionModel(4,1)
        learning_rate = 0.01
        self.optimizer11 = torch.optim.SGD(self.model11.parameters(), lr=learning_rate)
        self.optimizer21 = torch.optim.SGD(self.model21.parameters(), lr=learning_rate)
        self.max = 0
        self.min = 0
    def max_min(self,train_temp):
        max = np.max(train_temp,axis = 0)
        min = np.min(train_temp,axis = 0)
        return max,min
    def normalization(self,train_temp):
        for i in range(train_temp.shape[0]):
            train_temp[i,:] = (train_temp[i,:] -self.min) / (self.max-self.min)
        return train_temp
    def deal_data(self,filenametrain,filenameresult):
        self.train_temp = []
        with open(filenametrain) as tdata:
            while True:
                line = tdata.readline()
                if not line:
                    break
                self.train_temp.append([float(i) for i in line.split()])
        self.train_temp = np.array(self.train_temp)
        self.max,self.min = self.max_min(self.train_temp)
        self.normalization(self.train_temp)

        self.result_temp = []
        with open(filenameresult) as rdata:
            while True:
                line = rdata.readline()
                if not line:
                    break
                self.result_temp.append([float(i) for i in line.split()])
        self.result_temp = np.array(self.result_temp)

        self.x = np.mat(self.train_temp)
        self.x = torch.tensor(self.x).float()
        self.y = np.mat(self.result_temp)
        self.y = torch.tensor(self.y).float()
    def save(self,model,path,optimizer,epoch):
        state = {'model':model,'optimizer':optimizer,'epoch':epoch,'max':self.max,'min':self.min}
        torch.save(state,path)
    def load(self,model_number):
        if model_number == 1:
            checkpoint = torch.load('model1.pkl')
            self.model11.load_state_dict(checkpoint['model'])
            self.optimizer11.load_state_dict(checkpoint['optimizer'])
            self.epochs11 = checkpoint['epoch']
            self.max = checkpoint['max']
            self.min = checkpoint['min']
            #print(self.model1,self.optimizer1,self.epochs1)
        elif model_number == 2:
            checkpoint = torch.load('model2.pkl')
            self.model21.load_state_dict(checkpoint['model'])
            self.optimizer21.load_state_dict(checkpoint['optimizer'])
            self.epochs21 = checkpoint['epoch']
            self.max = checkpoint['max']
            self.min = checkpoint['min']
            #print(self.model2, self.optimizer2, self.epochs2)
        elif model_number == 3:
            checkpoint = torch.load('model3.pkl')
            self.model31.load_state_dict(checkpoint['model'])
            self.epochs31 = checkpoint['epoch']
            self.max = checkpoint['max']
            self.min = checkpoint['min']
    def model1(self):
        model = Model1.NonLinearRegression()
        criterion = nn.MSELoss()  # 损失函数为均方差
        learning_rate = 0.01
        self.optimizer11 = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.Draw = []

        for epoch in range(self.epochs11):
            epoch += 1
            # 清空梯度参数
            self.optimizer11.zero_grad()
            # 获得输出
            outputs = model(self.x)
            # 计算损失
            loss = criterion(outputs, self.y)
            self.Draw.append(loss)
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer11.step()
        return model

    def model2(self):
        input_dim = 4
        output_dim = 1
        model = Model2.LinearRegressionModel(input_dim, output_dim)
        criterion = nn.MSELoss()  # 损失函数为均方差

        learning_rate = 0.01
        self.optimizer21 = torch.optim.SGD(model.parameters(), lr=learning_rate)
        '''训练网络'''

        for epoch in range(self.epochs21):
            epoch += 1
            # 清空梯度参数
            self.optimizer21.zero_grad()
            # 获得输出
            outputs = model(self.x)
            # 计算损失
            loss = criterion(outputs, self.y)

            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer21.step()
        return model

    def model3(self):
        model = Model3.OurNeuralNetwork()
        model.train(self.train_temp, self.result_temp)
        return model

    def train_model(self,model_number):
        self.delete_tempfile("temp_result.txt")
        filenametrain = "train_data.txt"
        filenameResult = "result_data.txt"
        self.deal_data(filenametrain, filenameResult)
        if model_number == 1:
            self.epochs11 = 1000
            self.model11 = self.model1()
        elif model_number == 2:
            self.epochs21 = 1000
            self.model21 = self.model2()
        elif model_number == 3:
            self.epochs31 = 1000
            self.model31 = self.model3()

        pre = []
        real = []
        for i in range(self.train_temp.shape[0]):
            predict = np.mat(self.train_temp[i, :])
            predict = torch.tensor(predict).float()
            if model_number == 1:
                predict = self.model11(predict).data.numpy()
            elif model_number == 2:
                predict = self.model21(predict).data.numpy()
            elif model_number == 3:
                predict = np.apply_along_axis(self.model31.feedforward, 1, predict)
            predict = predict[0, :][0]
            train = self.result_temp[i, :][0]
            real.append(train)
            pre.append(predict)
            with open("temp_result.txt", 'a') as f:
                f.write(str(predict) + '\n')

        filenameResult = "temp_result.txt"
        self.deal_data(filenametrain, filenameResult)
        if model_number == 1:
            self.epochs12 = 1000
            self.model12 = self.model1()
        elif model_number == 2:
            self.epochs22 = 1000
            self.model22 = self.model2()
        elif model_number == 3:
            self.epochs32 = 1000
            self.model32  = self.model3()
        variance = 0
        var = []
        for i in range(self.train_temp.shape[0]):
            predict = np.mat(self.train_temp[i, :])
            predict = torch.tensor(predict).float()
            if model_number == 1:
                predict = self.model12(predict).data.numpy()
            elif model_number == 2:
                predict = self.model22(predict).data.numpy()
            elif model_number == 3:
                predict = np.apply_along_axis(self.model32.feedforward, 1, predict)
            predict = predict[0, :][0]
            train = self.result_temp[i, :][0]
            variance = math.pow((predict - train), 2) + variance
            var.append(math.pow((predict - train), 2))
        variance = variance / self.train_number
        print('v:',model_number,variance)
        self.z = [[5.000,0.300,0.500,10.000]]
        self.z = np.array(self.z)
        self.z = self.normalization(self.z)
        self.z = np.mat(self.z)
        self.z = torch.tensor(self.z).float()
        if model_number == 1:
            predict = self.model11(self.z).data.numpy()
            self.save(self.model11.state_dict(), 'model1.pkl',self.optimizer11.state_dict(),self.epochs11)
            self.var1 = var
            return self.var1
        elif model_number == 2:
            predict = self.model21(self.z).data.numpy()
            self.save(self.model21.state_dict(), 'model2.pkl', self.optimizer21.state_dict(), self.epochs21)
            self.var2 = var
            self.draw_var(self.var1,self.var2)
            return self.var2
        elif model_number == 3:
            predict = np.apply_along_axis(self.model31.feedforward, 1, self.z)
            # self.save(self.mode131.state_dict(), 'model3.pkl', "none", self.epochs31)
            self.var3 = var
            self.draw_var3(self.var3)
            return self.var3
        self.delete_tempfile("temp_result.txt")


    def delete_tempfile(self,filename):
        my_file = filename  # 文件路径
        if os.path.exists(my_file):  # 如果文件存在
            # 删除文件，可使用以下两种方法。
            os.remove(my_file)  # 则删除
            # os.unlink(my_file)
        else:
            print('no such file:%s' % my_file)

    def draw_diff(self,pre,real,title):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(np.arange(200),real, "bs")
        plt.plot(np.arange(200), pre, "r^")
        plt.title(title + "：红色为预测，蓝色为真实值")
        plt.ylim(10,80)
        plt.show()

    def draw_var(self,var1,var2):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(np.arange(200), var1, "bs")
        plt.plot(np.arange(200), var2, "r^")
        plt.title("红色为使用线性，蓝色为使用非线性")
        plt.ylim(0,0.8)
        plt.show()

    def draw_var3(self,var3):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(np.arange(200), var3, "bs")
        plt.title("自拟")
        plt.ylim(0,1)
        plt.show()

    def update(self,model_number,x,y):
        self.load(model_number)
        max,min = self.max_min(x)
        for i in range(len(max)):
            if max[i] > self.max[i]:
                self.max[i] = max[i]
            if min[i] < self.min[i]:
                self.min[i] = min[i]
        x = self.normalization(x)
        x = np.mat(x)
        x = torch.tensor(x).float()
        if model_number == 1:
            criterion = nn.MSELoss()
            for epoch in range(self.epochs11+1,self.epochs11+500):
                epoch += 1
                # 清空梯度参数
                self.optimizer11.zero_grad()
                # 获得输出
                outputs = self.model11(x)
                # 计算损失
                loss = criterion(outputs, y)
                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer11.step()
            self.epochs11 = self.epochs11 + 500
            self.save(self.model11.state_dict(), 'model1.pkl', self.optimizer11.state_dict(), self.epochs11)
        elif model_number == 2:
            criterion = nn.MSELoss()  # 损失函数为均方差
            for epoch in range(self.epochs21+1,self.epochs21+500):
                epoch += 1
                # 清空梯度参数
                self.optimizer21.zero_grad()
                # 获得输出
                outputs = self.model21(x)
                # 计算损失
                loss = criterion(outputs, y)
                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer21.step()
            self.epochs21 = self.epochs21 + 500
            self.save(self.model21.state_dict(), 'model2.pkl', self.optimizer21.state_dict(), self.epochs21)

    def predict(self,model_number):
        self.load(1)
        self.load(2)
        self.precdict_temp = []
        pre = []
        real = []
        with open("train_data.txt") as pdata:
            while True:
                line = pdata.readline()
                if not line:
                    break
                self.precdict_temp.append([float(i) for i in line.split()])
        with open("result_data.txt") as pdata:
            while True:
                line = pdata.readline()
                if not line:
                    break
                for i in line.split():
                    real.append((float)(i))
        self.precdict_temp = np.array(self.precdict_temp)
        self.precdict_temp = self.normalization(self.precdict_temp)     #数据预处理
        self.precdict_temp = np.mat(self.precdict_temp)
        self.precdict_temp = torch.tensor(self.precdict_temp).float()

        if model_number==1:
            self.predict_result1 = self.model11(self.precdict_temp).data.numpy()
            for i in range(len(self.predict_result1)):
                pre.append(self.predict_result1[i,:][0])
            self.draw_diff(pre,real,"非线性")
            pre.save('predict_result1.txt')

        elif model_number == 2:
            pre = []
            self.predict_result2 = self.model21(self.precdict_temp).data.numpy()
            for i in range(len(self.predict_result2)):
                pre.append(self.predict_result2[i,:][0])
            self.draw_diff(pre, real,"线性")
            pre.save('predict_result2.txt')

        elif model_number == 3:
            self.predict_result3 = np.apply_along_axis(self.model31.feedforward, 1, self.precdict_temp)
            pre = []
            for i in range(len(self.predict_result3)):
                pre.append(self.predict_result3[i, :][0])
            self.draw_diff(pre, real, "自定义")
            pre.save('predict_result3.txt')


# if __name__ == '__main__':
#     run = train()
# #     #run.deal_data("train_data.txt","result_data.txt")
# #     # run.train_model(1)
# #     # run.train_model(2)
#     run.train_model(3)
# #     run.predict(3)