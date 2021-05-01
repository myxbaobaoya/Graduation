import keras
import numpy as np
import torch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# Sequential按顺序构成的模型
from keras.models import Sequential
# Dense全连接层
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# # 使用Numpy生成200个-0.5~0.5之间的值
# x_data = np.linspace(-0.5, 0.5, 200)
# noise = np.random.normal(0, 0.02, x_data.shape)
#
# # y_data= x_data**2 + noise
# y_data = np.square(x_data) + noise  # 效果与上面一致
#
# # 显示随机点
# plt.scatter(x_data, y_data)
# plt.show()


train_temp=[]
with open("train_data.txt") as tdata:
    while True:
        line=tdata.readline()
        temp = np.zeros((3,32,32))
        if not line:
            break
        j = 0
        for i in line.split():
            temp[0][j][0] = float(i)
            j = j+1
        train_temp.append(temp)
train_temp = np.array(train_temp)

result_temp=[]
with open("result_data.txt") as rdata:
    while True:
        line=rdata.readline()
        if not line:
            break
        for i in line.split():
            result_temp.append(float(i))
print(result_temp)


import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x



net = LeNet()
#loss_function = nn.CrossEntropyLoss()
loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
x = torch.tensor(train_temp).float()
y = torch.tensor(result_temp).float()
print(x[0])
print(y)
x = x.squeeze(-1)

for epoch in range(5):  # loop over the dataset multiple times
    # for i in range(len(train_temp)):

        optimizer.zero_grad()
        # 获得输出
        outputs = net(x)
        # 计算损失

        loss = loss_function(outputs, y)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        print(outputs)

# predict = net(x).data.numpy()
outputs = net(x).data.numpy()
#predict = torch.max(outputs, dim=1)[1].data.numpy()
print(outputs)


    # running_loss = 0.0
    # for step, data in enumerate(train_temp, start=0):
    #     # get the inputs; data is a list of [inputs, labels]
    #     inputs, labels = data
    #
    #     # zero the parameter gradients
    #     optimizer.zero_grad()
    #     # forward + backward + optimize
    #     outputs = net(inputs)
    #     loss = loss_function(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    #
    #     # print statistics
    #     running_loss += loss.item()
    #     if step % 500 == 499:    # print every 500 mini-batches
    #         with torch.no_grad():
    #             outputs = net(val_image)  # [batch, 10]
    #             predict_y = torch.max(outputs, dim=1)[1]
    #             accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
    #
    #             print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
    #                   (epoch + 1, step + 1, running_loss / 500, accuracy))
    #             running_loss = 0.0

print('Finished Training')

save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)
# import os
# import json
#
# import torch
# import torch.nn as nn
# from torchvision import transforms, datasets
# import torch.optim as optim
# from tqdm import tqdm
#
# from VGGnet import vgg
#
#
# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("using {} device.".format(device))
#
#     data_transform = {
#         "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
#         "val": transforms.Compose([transforms.Resize((224, 224)),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
#
#     data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
#     image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
#     assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
#     train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
#                                          transform=data_transform["train"])
#     train_num = len(train_dataset)
#
#     # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
#     flower_list = train_dataset.class_to_idx
#     cla_dict = dict((val, key) for key, val in flower_list.items())
#     # write dict into json file
#     json_str = json.dumps(cla_dict, indent=4)
#     with open('class_indices.json', 'w') as json_file:
#         json_file.write(json_str)
#
#     batch_size = 32
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#     print('Using {} dataloader workers every process'.format(nw))
#
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=batch_size, shuffle=True,
#                                                num_workers=nw)
#
#     validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
#                                             transform=data_transform["val"])
#     val_num = len(validate_dataset)
#     validate_loader = torch.utils.data.DataLoader(validate_dataset,
#                                                   batch_size=batch_size, shuffle=False,
#                                                   num_workers=nw)
#     print("using {} images for training, {} images for validation.".format(train_num,
#                                                                            val_num))
#
#     # test_data_iter = iter(validate_loader)
#     # test_image, test_label = test_data_iter.next()
#
#     model_name = "vgg16"
#     net = vgg(model_name=model_name, num_classes=5, init_weights=True)
#     net.to(device)
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr=0.0001)
#
#     epochs = 30
#     best_acc = 0.0
#     save_path = './{}Net.pth'.format(model_name)
#     train_steps = len(train_loader)
#     for epoch in range(epochs):
#         # train
#         net.train()
#         running_loss = 0.0
#         train_bar = tqdm(train_loader)
#         for step, data in enumerate(train_bar):
#             images, labels = data
#             optimizer.zero_grad()
#             outputs = net(images.to(device))
#             loss = loss_function(outputs, labels.to(device))
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#
#             train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
#                                                                      epochs,
#                                                                      loss)
#
#         # validate
#         net.eval()
#         acc = 0.0  # accumulate accurate number / epoch
#         with torch.no_grad():
#             val_bar = tqdm(validate_loader)
#             for val_data in val_bar:
#                 val_images, val_labels = val_data
#                 outputs = net(val_images.to(device))
#                 predict_y = torch.max(outputs, dim=1)[1]
#                 acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
#
#         val_accurate = acc / val_num
#         print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
#               (epoch + 1, running_loss / train_steps, val_accurate))
#
#         if val_accurate > best_acc:
#             best_acc = val_accurate
#             torch.save(net.state_dict(), save_path)
#
#     print('Finished Training')
#
#
# if __name__ == '__main__':
#     main()
# # 建立一个顺序模型
# model = Sequential()
# # 1-10-1: 加入一个隐藏层（10个神经元）：来拟合更加复杂的线性模型。添加激活函数，来计算函数的非线性
#
# model.add(Dense(units=10, input_dim=4, activation='relu'))  # 全连接层：输入一维数据，输出10个神经元
# # model.add(Activation('tanh')) # 也可以直接在Dense里面加激活函数
# model.add(Dense(units=1, activation='tanh'))  # 全连接层：由于有上一层的添加，所以输入维度默认是10（可以不用写），输出1个值（要写）
# # model.add(Activation('tanh'))
#
#
# # 自定义优化器SDG , 学习率默认是0.01(太小，导致要迭代好多次才能较好的拟合数据)
# sgd = SGD(lr=0.01)
# model.compile(optimizer=sgd, loss='mse')
#
# # 训练3000次数据
# for step in range(3001):
#     cost = model.train_on_batch(train_temp, result_temp)
#     if step % 500 == 0:
#         print('cost: ', cost)
#
# # x_data输入神经网络中，得到预测值y_pred
# z = np.mat('0.060 0.300 0.100 6.000')
# y_pred = model.predict(train_temp)
# print(y_pred)
#
# # # 显示随机点
# # plt.scatter(x_data, y_data)
# # plt.plot(x_data, y_pred, 'r-', lw=3)
# # plt.show()
