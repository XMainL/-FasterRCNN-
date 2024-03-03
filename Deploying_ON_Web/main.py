"""
FileName : Main
Target   : 实现肺炎患病与非患病情况的预测
Steps    :
-- 1. 建立训练数据集
-- 2. 建立验证和测试数据集
-- 3. 定义网络
-- 4. 定义损失函数
-- 5. 定义优化器
-- 6. 传入参数
-- 7. 运用模型进行预测
"""


# 导入需要的包
from matplotlib.testing.jpl_units import Epoch
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import utils
from torchvision import datasets
import matplotlib.pyplot as plt
from torch import nn
from torch.utils import data
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
from model import ResNet50
from tqdm import tqdm


# 图像预处理
# Resize(): 将传入的图像重新定义为 500 * 500 大小
# RandomHorizontalFlip(): 随机翻转图像 作为图像增强
# ToTensor(): 将图片转化为张量的格式
# Normalize(): 用均值和标准差对图片进行归一化
transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
                                ])

# 定义分类列表
# 0: 定义为未发现患病
# 1: 定义为发现肺炎症状
class_list = {0: "NotFinding", 1: "Pneumonia"}  # 用于后续预测的时候可以使用，用预测到的标签来直接获取相应的类别

"""弃用代码--改用ImageFolder进行数据读取"""
# 定义 ChestXray14类
# 包括 __init__ 构造函数
# 包括 __getitem__ __len__ 的重写
# class ChestXray14(Dataset):
#     def __init__(self, list_IDs, labels, mode):
#         super(ChestXray14, self).__init__()
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.mode = mode
#
#     def __len__(self):
#         """Denotes the total number of samples"""
#         return self.list_IDs
#
#     def __getitem__(self, index):
#         """Generates one sample of data"""
#         # Select sample
#         ID = self.list_IDs[index]
#         # Load data and get label
#         X = torch.load('PictureClassifar/dataset/' + self.mode + ID + '.jpg')
#         y = self.labels[ID]
#         return X, y


# 定义网络
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
              'M'],
}


def vgg(model_name="vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model


# 定义参数列表
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.01

# 定义训练函数
# 传入数据 模型 损失函数 优化器
model = ResNet50.ResNet50().to(device)
# model = vgg("vgg16").to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# 学习率每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# 中断训练重新加载模型
last_epoch = 10
# model = torch.load('model_name.pth')
# optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': 0.01}], lr=learning_rate)
# lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=last_epoch)


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, Epoch, EPOCHS, BATCH_SIZE):
    """
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param Epoch:
    :param EPOCHS:
    :param BATCH_SIZE:
    :return:
    """
    model.train()
    loss, current, n = 0.0, 0.0, 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    right = 0
    for batch_train, (batch_x, batch_y) in loop:
        image, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, batch_y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(batch_y == pred) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

        # 累加识别正确的样本数
        right += (pred == batch_y).sum()

        # 更新信息
        loop.set_description(f'Epoch [{Epoch}/{EPOCHS}]')
        loop.set_postfix(loss=loss/(batch_train+1),
                         acc=float(right)/float(BATCH_SIZE * batch_train+len(image)))

    train_loss = loss / n
    train_acc = current / n
    # print('+------------+--------------+')
    # print('| train_loss |  ' + str(format(train_loss, '.8f') + '  |'))
    # print('| train_acc  |  ' + str(format(train_acc, '.8f') + '  |'))
    # print('+------------+--------------+')
    return train_loss, train_acc


# def train(dataloader, model, loss_fn, optimizer):
#     loss, current, n = 0.0, 0.0, 0
#     for batch, (x, y) in enumerate(dataloader):
#         image, y = x.to(device), y.to(device)
#         output = model(image)
#         cur_loss = loss_fn(output, y)
#         _, pred = torch.max(output, axis=1)
#         cur_acc = torch.sum(y==pred) / output.shape[0]
#
#         # 反向传播
#         optimizer.zero_grad()
#         cur_loss.backward()
#         optimizer.step()
#         loss += cur_loss.item()
#         current += cur_acc.item()
#         n = n+1
#
#     train_loss = loss / n
#     train_acc = current / n
#     print('+------------+--------------+')
#     print('| train_loss |  ' + str(format(train_loss, '.8f') + '  |'))
#     print('| train_acc  |  ' + str(format(train_acc, '.8f') + '  |'))
#     print('+------------+--------------+')
#     return train_loss, train_acc


def test(dataloader, model, loss_fn, Epoch, EPOCHS, BATCH_SIZE):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    right = 0
    with torch.no_grad():
        for batch_train, (batch_x, batch_y) in loop:
            image, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, batch_y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(batch_y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

            # 累加识别正确的样本数
            right += (pred == batch_y).sum()

            # 更新信息
            loop.set_description(f'Epoch [{Epoch}/{EPOCHS}]')
            loop.set_postfix(loss=loss / (batch_train + 1),
                             acc=float(right) / float(BATCH_SIZE * batch_train + len(image)))

    test_loss = loss / n
    test_acc = current / n
    # print('+------------+--------------+')
    # print('| train_loss |  ' + str(format(train_loss, '.8f') + '  |'))
    # print('| train_acc  |  ' + str(format(train_acc, '.8f') + '  |'))
    # print('+------------+--------------+')
    return test_loss, test_acc


# 定义一个验证函数
# def test(dataloader, model, loss_fn):
#     # 将模型转化为验证模型
#     model.eval()
#     loss, current, n = 0.0, 0.0, 0
#     with torch.no_grad():
#         for batch, (x, y) in enumerate(dataloader):
#             image, y = x.to(device), y.to(device)
#             output = model(image)
#             cur_loss = loss_fn(output, y)
#             _, pred = torch.max(output, axis=1)
#             cur_acc = torch.sum(y == pred) / output.shape[0]
#             loss += cur_loss.item()
#             current += cur_acc.item()
#             n = n + 1
#
#     test_loss = loss / n
#     test_acc = current / n
#     # 绘制TestData框
#     print('| test_loss  |  ' + str(format(test_loss, '.8f') + '  |'))
#     print('| test_acc   |  ' + str(format(test_acc , '.8f') + '  |'))
#     print('+------------+--------------+')
#     return test_loss, test_acc


# 定义画图函数
def matplot_loss(train_loss, test_loss):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和测试集loss值对比")
    plt.show()


def matplot_acc(train_acc, test_acc):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(train_acc, label='train_acc')
    plt.plot(test_acc, label='test_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("Train Accuracy Contrast With Test Accuracy")
    plt.title("训练集和测试集Accuracy对比")
    plt.show()


# 实例化Dataset 和 Dataloader对象
# Generators
params = {
    'batch_size': 32,
    'shuffle': False,
}

# 载入训练集和测试集b
# train_datas = ChestXray14(partition['train'], labels, 'train')
# train_loader = data.DataLoader(train_datas, **params)
#
# test_datas = ChestXray14(partition['test'], labels, 'test')
# test_loader = data.DataLoader(test_datas, **params)
train_root = "C:/Users/XMainL/PycharmProjects/DeepLearning/PictureClassifar/test_dataset/train"
test_root = "C:/Users/XMainL/PycharmProjects/DeepLearning/PictureClassifar/test_dataset/val"
train_dataset = ImageFolder(train_root, transform=transform)
test_dataset = ImageFolder(test_root, transform=transform)
# 定义BATCH_SIZE 方便后续使用
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 开始训练
loss_train = []
acc_train = []
loss_test = []
acc_test = []
start_epoch = 1
epoch = 20
min_acc = 0


# 中断训练解除下面一行注释
# for t in range(last_epoch + 1, epoch):
for t in range(epoch):
    lr_scheduler.step()
    print("\033[31m+···························+\033[0m")
    print('|' + f"Epoch{t+1}".center(27) + '|')
    train_loss, train_acc = train(train_loader, model, loss_function, optimizer, Epoch=t+1, EPOCHS=epoch, BATCH_SIZE=batch_size)
    test_loss, test_acc = test(test_loader, model, loss_function, Epoch=t+1, EPOCHS=epoch,BATCH_SIZE=batch_size)
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_test.append(test_loss)
    acc_test.append(test_acc)
    # 保存最好的模型权重
    if test_acc > min_acc:
        save_folder = 'save_model'
        if not os.path.exists(save_folder):
            os.mkdir('save_model')
        min_acc = test_acc

        print('|  ALREADY SAVE BEST MODEL  |')
        print('+---------------------------+')
        torch.save(model, 'save_model/best_model.pth')

    # 每隔指定次数保存一次模型
    save_num = 5
    if (t+1) % save_num == 0:
        torch.save(model, f'log/model_epoch{t+1}')

    # 保存最后一轮的权重文件
    if t == epoch-1:
        torch.save(model, 'save_model/last_model.pth')

    Epoch = start_epoch + 1


matplot_loss(loss_train, loss_test)
matplot_acc(acc_train, acc_test)
print()
print('\033[32mALREADY DONE!\033[0m')


# 读取模型
# model = torch.load('model_name.pth')



