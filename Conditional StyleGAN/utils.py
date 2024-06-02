import torch
import random
import math
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# Generator and Discriminator Utility Functions
def initialize_linear_layer(layer):
    nn.init.xavier_normal(layer.weight)
    layer.bias.data.zero_()

def initialize_conv_layer(layer):
    nn.init.kaiming_normal(layer.weight)
    if layer.bias is not None:
        layer.bias.data.zero_()
        
def equal_layer(module, name="weight"):
    EqualLR.apply(module, name)
    return module


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        input = weight.data.size(1) * weight.data[0][0].numel()
        return weight * math.sqrt(2 / input)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

class EqualConvLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_layer(conv)

    def forward(self, x):
        return self.conv(x)

class EqualLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        linear = nn.Linear(in_channels, out_channels)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = equal_layer(linear)

    def forward(self, x):
        return self.linear(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, kernel_size2=None, padding2=None):
        super().__init__()
        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv_block = nn.Sequential(
            EqualConvLayer(in_channels, out_channels, kernel1, pad1),
            nn.LeakyReLU(0.2),
            EqualConvLayer(out_channels, out_channels, kernel2, pad2),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv_block(x)

class AdaIN(nn.Module):
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.adain = nn.InstanceNorm2d(in_channels)
        self.style = EqualLinearLayer(style_dim, in_channels*2)
        self.style.linear.bias.data[:in_channels] = 1
        self.style.linear.bias.data[in_channels:] = 0

    def forward(self, x, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        x = self.adain(x)
        x = gamma*x + beta
        return x

class AddNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise):
        return x + self.weight*noise

class ConstantIn(nn.Module):
    def __init__(self, channels, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channels, size, size))

    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out

class StyleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, style_dim=512, initial=False):
        super().__init__()
        if initial:
            self.conv1 = ConstantIn(in_channels)
        else:
            self.conv1 = EqualConvLayer(in_channels, out_channels, kernel_size, padding=padding)
        self.noise1 = equal_layer(AddNoise(out_channels))
        self.adain1 = AdaIN(out_channels, style_dim)
        self.leakyrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConvLayer(out_channels, out_channels, kernel_size, padding=padding)
        self.noise2 = equal_layer(AddNoise(out_channels))
        self.adain2 = AdaIN(out_channels, style_dim)
        self.leakyrelu2 = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise):
        x = self.conv1(x)
        x = self.noise1(x, noise)
        x = self.adain1(x, style)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.noise2(x, noise)
        x = self.adain2(x, style)
        x = self.leakyrelu2(x)
        return x
    

# Training Utility Functions
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def get_data_sample(dataset, batch_size, image_size=4):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset.transform = transform
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16)

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr * group.get('mult', 1)
