""" Define the CNN architecture, based on torchvison's VGG net.
"""
import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, ftrsC, features, init_weights=True):
        super(VGG, self).__init__()
        self.ftrsC = ftrsC
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


DEFAULT = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512]
def net(batch_norm, cfg=DEFAULT, **kwargs):
    model = VGG(cfg[-1], make_layers(cfg , batch_norm=batch_norm), **kwargs)
    return model
