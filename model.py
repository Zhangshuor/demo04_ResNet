import torch.nn as nn
import torch


# 首先定义18，34层用的残差结构（实线结构和虚线结构都要有）
class BasicBlock(nn.Module):
    # 主分支的卷积核个数没有变化
    expansion = 1
    # downsample=None对应虚线残差结构

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # stride=1:output = (input-3+2*1)/1+1=input
        # stride=2:output = (input-3+2*1)/2+1=input/2+0.5=input/2(向下取整)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # BN在卷积层和激活层之间
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # 无论实线还是虚线结构，stride=1
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        # 实线直接把x那过去，跳过
        identity = x
        # 如果虚线残差结构，走对应的下采样函数
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# 定义50，101,152层用的残差结构（实线结构和虚线结构都要有）


class Bottleneck(nn.Module):
    # 主分支的卷积核个数有一个4倍的增加
    expansion = 4
    # downsample=None对应虚线残差结构

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        # BN在卷积层和激活层之间
        self.bn1 = nn.BatchNorm2d(out_channel)
        # ---------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        # 实线直接把x那过去，跳过
        identity = x
        # 如果虚线残差结构，走对应的下采样函数
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        # output = (input-7+3*2)/2+1=input/2+0.5=input/2(高宽减半)
        self.conv1 = nn.Conv2d(
            3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # output = (input-3+1*2)/2+1=input/2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block=block, channel = 64, block_num = block_num[0])
        self.layer2 = self._make_layer(block=block, channel = 128, block_num = block_num[1], stride=2)
        self.layer3 = self._make_layer(block=block, channel = 256, block_num = block_num[2], stride=2)
        self.layer4 = self._make_layer(block=block, channel = 512, block_num = block_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # 定义制作残差快层的函数
    '''
    block:BasicBlock/Bottleneck
    channel:每个残差块第一层的channel
    block_num:该层一共多少个残差结构
    '''

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        # 残差块的第一层
        layers.append(block(self.in_channel,channel,downsample=downsample,stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channel,channel))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet34(num_classes=1000,include_top=True):
    return ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes,include_top=include_top)

def resnet101(num_classes=1000,include_top=True):
    return ResNet(Bottleneck,[3,4,23,3],num_classes=num_classes,include_top=include_top)