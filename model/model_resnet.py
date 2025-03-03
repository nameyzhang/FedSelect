import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, self.expansion * out_planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, out_planes, stride = 1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_planes, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, latent_output=False):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)

        x1 = self.linear(out)

        if latent_output == False:
            output = x1
        else:
            output = out
        return output




def ResNet18(num_classes = 10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# def ResNet34(num_classes = 10):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def ResNet50(num_classes = 14):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def ResNet101(num_classes = 10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def ResNet152(num_classes = 10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class ResNet32(nn.Module):
    def __init__(self, args, num_classes, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        a = 1
        if args.dataset_name == 'CIFAR10' or args.dataset_name == 'CIFAR100':
            a = 3
        elif args.dataset_name == 'MNIST':
            a = 1
        self.in_planes = 16

        self.conv1 = nn.Conv2d(a, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("x", x)
        # print("x.size()", x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):


    def __init__(self, num_classes=100):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make_layer(16, 16, 3)
        self.layer2 = self._make_layer(16, 32, 4, stride=1)
        self.layer3 = self._make_layer(32, 64, 6, stride=1)
        self.layer4 = self._make_layer(64, 64, 3, stride=1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 1, stride, bias=False), nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)