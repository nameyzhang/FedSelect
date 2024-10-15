import torch
from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

        # 卷积层 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),          # 批归一化
            nn.ReLU(),
        )
        # 下采样
        self.subsample1 = nn.MaxPool2d(kernel_size=2, stride=2)     # 最大池化
        # 卷积层 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # 下采样
        self.subsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接
        self.L1 = nn.Linear(16 * 4 * 4, 120)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.L3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.subsample1(out)
        out = self.layer2(out)
        out = self.subsample2(out)
        # 将上一步输出的16个 5*5 特征图中的 400个像素展平成一维向量, 以便下一步全连接
        out = out.view(out.size(0), -1)

        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu1(out)
        out = self.L3(out)

        return out



# class LeNet(nn.Module):                         # LeNet 类继承自 nn.Module, 用于构建神经网络
#     def __init__(self, input_channels=3):       # __init__ 方法是构造函数, 用于初始化网络的层
#         super(LeNet, self).__init__()           # 调用父类 nn.Module 的构造函数
#         self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)   # 卷积核大小为 5*5
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.batch_norm = nn.BatchNorm2d(16)    # 定义了批量归一化层, 对16个通道进行归一化
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 定义了第一个全连接层, 从卷积成展平后的特征图大小映射到120个神经元
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):                       # forward 方法定义了前向传播的计算过程; 输入 x 是一个张量, 表示输入图像
#         x = F.relu(self.conv1(x))               # 对输入的录像应用第一个卷积层, 然后应用 ReLU 激活函数
#         x = F.max_pool2d(x, 2)        # 对输出应用 2 * 2 最大池化层, 减少特征图的大小
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = self.batch_norm(x)                  # 对池化后的特征图应用批量归一化层
#         feature = x.view(x.size(0), -1)         # 将特征图展平为二维向量, 展平后的大小为 (batch_size, 16 * 5 *5)
#         x = F.relu(self.fc1(feature))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#
#         return x, feature                       # 返回分类结果 x 和展平的特征图 feature
