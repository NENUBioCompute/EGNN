import torch.nn as nn
import torch
import math
from Fusion import MultiModalFusion
def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, (3,1), padding=(1,0), )  


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, (1, 1), )


def norm_layer(planes):
    return nn.BatchNorm2d(planes)
class eca_layer(nn.Module):

    def __init__(self, channel, gamma=2, b=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) /gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.b = b

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Block(nn.Module):
    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d((2, 1),(2, 1))



    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)
        return out


class ResNet8(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.inplanes = 1

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])
        self.eca = eca_layer(512)
        self.out_dim = channels[3]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 2))


        self.fc = nn.Sequential(
            nn.Linear(self.out_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.sigmoid = nn.Sigmoid()
        # Multi-scale information fusion
        self.fusion = MultiModalFusion(512, 8, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.eca(x)
        
        flattened_features = torch.flatten(x, 1) 
        x = self.fc(flattened_features)
        output = self.sigmoid(x)
        return output, flattened_features


def resnet8():
    return ResNet8([64, 128, 256, 512])


if __name__ == '__main__':
    model = resnet8()
    import torch

    x = torch.ones([16, 1, 180, 2])
    y = model(x)
    print(y.shape)
