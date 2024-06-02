import torch
from torch import nn
import torch.nn.functional as F
from utils import ConvBlock, EqualConvLayer, EqualLinearLayer

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, condition=2):
        super().__init__()
        if type(condition) not in (list, tuple):
            condition = [condition]
        condition_channels = [num_classes if i in condition else 0 for i in range(9)]
        self.progression_list = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1),
                ConvBlock(32, 64, 3, 1),
                ConvBlock(64, 128, 3, 1),
                ConvBlock(128, 256, 3, 1),
                ConvBlock(256, 512, 3, 1),
                ConvBlock(512, 512, 3, 1),
                ConvBlock(512, 512, 3, 1),
                ConvBlock(512, 512, 3, 1),
                ConvBlock(512, 512, 3, 1, 4, 0),
            ]
        )
        self.rgb_list = nn.ModuleList(
            [
                EqualConvLayer(3+condition_channels[8], 16, 1),
                EqualConvLayer(3+condition_channels[7], 32, 1),
                EqualConvLayer(3+condition_channels[6], 64, 1),
                EqualConvLayer(3+condition_channels[5], 128, 1),
                EqualConvLayer(3+condition_channels[4], 256, 1),
                EqualConvLayer(3+condition_channels[3], 512, 1),
                EqualConvLayer(3+condition_channels[2], 512, 1),
                EqualConvLayer(3+condition_channels[1], 512, 1),
                EqualConvLayer(3+condition_channels[0], 512, 1),
            ]
        )
        self.num_layers = len(self.progression_list)
        self.equal_linear = EqualLinearLayer(512, 1)
        self.condition = condition
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)

    def forward(self, x, label, step=0, alpha=-1):
        label = self.label_emb(label).view(-1, self.num_classes, 1, 1)

        for i in range(step, -1, -1):
            index = self.num_layers - i - 1
            downsample_input = current_input = x
            if i in self.condition:
                current_input = torch.cat([x, label.repeat(1, 1, *x.shape[2:])], dim=1)
            if i-1 in self.condition:
                downsample_input = torch.cat([x, label.repeat(1, 1, *x.shape[2:])], dim=1)
            if i == step:
                out = self.rgb_list[index](current_input)
            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression_list[index](out)
            if i > 0:
                out = F.interpolate(out, scale_factor=0.5, mode="bilinear", align_corners=False)

                if i == step and 0 <= alpha < 1:
                    skip_rgb = self.rgb_list[index + 1](downsample_input)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=0.5, mode="bilinear", align_corners=False)
                    out = (1 - alpha)*skip_rgb + alpha*out

        out = out.squeeze(2).squeeze(2)
        out = self.equal_linear(out)
        return out
    