import random
import torch
from torch import nn
import torch.nn.functional as F
from utils import StyleConvBlock, EqualConvLayer, EqualLinearLayer, PixelNorm

class Generator(nn.Module):
    def __init__(self, code_dim, num_classes=10, condition=2):
        super().__init__()
        if type(condition) not in (list, tuple):
            condition = [condition]
        self.progression_list = nn.ModuleList(
            [
                StyleConvBlock(512, 512, 3, 1, initial=True),
                StyleConvBlock(512, 512, 3, 1),
                StyleConvBlock(512, 512, 3, 1),
                StyleConvBlock(512, 512, 3, 1),
                StyleConvBlock(512, 256, 3, 1),
                StyleConvBlock(256, 128, 3, 1),
                StyleConvBlock(128, 64, 3, 1),
                StyleConvBlock(64, 32, 3, 1),
                StyleConvBlock(32, 16, 3, 1),
            ]
        )
        self.rgb_list = nn.ModuleList(
            [
                EqualConvLayer(512, 3, 1),
                EqualConvLayer(512, 3, 1),
                EqualConvLayer(512, 3, 1),
                EqualConvLayer(512, 3, 1),
                EqualConvLayer(256, 3, 1),
                EqualConvLayer(128, 3, 1),
                EqualConvLayer(64, 3, 1),
                EqualConvLayer(32, 3, 1),
                EqualConvLayer(16, 3, 1),
            ]
        )
        self.condition = condition
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)

    def forward(self, style, label, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]
        if len(style) < 2:
            inject_index = [len(self.progression_list) + 1]
        else:
            inject_index = random.sample(list(range(step)), len(style) - 1)

        crossover = 0
        for i, (conv, to_rgb) in enumerate(zip(self.progression_list, self.rgb_list)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))
                style_step = style[crossover]
            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]
                else:
                    style_step = style[0]

            style_step = style_step.clone()
            if i in self.condition:
                style_step[:, :self.num_classes] = self.label_emb(label).view(-1, self.num_classes)
            else:
                style_step[:, :self.num_classes] = torch.zeros(style_step.shape[0], self.num_classes).to(style_step.device)

            if i > 0 and step > 0:
                upsample = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
                out = conv(upsample, style_step, noise[i])
            else:
                out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)
                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.rgb_list[i - 1](upsample)
                    out = (1 - alpha)*skip_rgb + alpha*out
                break

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, num_mlp=8, num_classes=10, condition=2):
        super().__init__()
        self.generator = Generator(code_dim, num_classes=num_classes, condition=condition)
        layers = [PixelNorm()]
        for _ in range(num_mlp):
            layers.append(EqualLinearLayer(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*layers)

    def forward(self, x, label, noise=None, step=0, alpha=-1, mean_style=None, style_weight=0, mixing_range=(-1, -1),):
        styles = []
        if type(x) not in (list, tuple):
            x = [x]

        for i in x:
            styles.append(self.style(i))

        batch = x[0].shape[0]
        if noise is None:
            noise = []
            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=x[0].device))

        if mean_style is not None:
            styles_norm = []
            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))
            styles = styles_norm

        return self.generator(styles, label, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, x):
        style = self.style(x).mean(0, keepdim=True)
        return style
