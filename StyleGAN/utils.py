import torch
from torch import nn
import torch.nn.functional as F

factors = [1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.slope = (2/in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x*self.slope) + self.bias
    
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x/torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
    
class MappingNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.map = nn.Sequential(
            PixelNorm(),
            LinearLayer(in_features, out_features),
            nn.ReLU(),
            LinearLayer(out_features, out_features),
            nn.ReLU(),
            LinearLayer(out_features, out_features),
            nn.ReLU(),
            LinearLayer(out_features, out_features),
            nn.ReLU(),
            LinearLayer(out_features, out_features),
            nn.ReLU(),
            LinearLayer(out_features, out_features),
            nn.ReLU(),
            LinearLayer(out_features, out_features),
            nn.ReLU(),
            LinearLayer(out_features, out_features)
        )

    def forward(self, x):
        return self.map(x)
    
class AdaIN(nn.Module):
    def __init__(self, n_channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(n_channels)
        self.style_slope = LinearLayer(w_dim, n_channels)
        self.style_bias = LinearLayer(w_dim, n_channels)

    def forward(self, x, w):
        slope = self.style_slope(w).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        x = self.instance_norm(x)
        
        return (slope*x) + bias
    
class Noise(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.noise_weight = nn.Parameter(torch.zeros(1, n_channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.noise_weight + noise
    
class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.slope = ((2/in_channels)**0.5)/kernel_size
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x*self.slope) + self.bias.view(1, self.bias.shape[0], 1, 1)
    
class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv1 = ConvolutionLayer(in_channels, out_channels)
        self.noise1 = Noise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.conv2 = ConvolutionLayer(out_channels, out_channels)
        self.noise2 = Noise(out_channels)
        self.adain2 = AdaIN(out_channels, w_dim)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.leakyrelu(x)
        x = self.adain1(x, w)
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.leakyrelu(x)
        x = self.adain2(x, w)

        return x
    
class Generator(nn.Module):
    def __init__(self, in_channels, z_dim, w_dim, img_channels=3):
        super().__init__()
        self.map = MappingNetwork(z_dim, w_dim)
        self.start = nn.Parameter(torch.ones(1, in_channels, 4, 4))
        self.noise1 = Noise(in_channels)
        self.adain1 = AdaIN(in_channels, w_dim)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.noise2 = Noise(in_channels)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.adain2 = AdaIN(in_channels, w_dim)
        self.rgb = ConvolutionLayer(in_channels, img_channels, 1, 1, 0)
        self.progressive_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([self.rgb])

        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels*factors[i])
            conv_out_channels = int(in_channels*factors[i+1])
            self.progressive_blocks.append(GeneratorBlock(conv_in_channels, conv_out_channels, w_dim))
            self.rgb_layers.append(ConvolutionLayer(conv_out_channels, img_channels, 1, 1, 0))

    def fade(self, alpha, upscaled, generated):
        return torch.tanh(alpha*generated + (1-alpha)*upscaled)
    
    def forward(self, noise, alpha, steps):
        w = self.map(noise)
        x = self.adain1(self.noise1(self.start), w)
        x = self.conv(x)
        out = self.leakyrelu(self.noise2(x))
        out = self.adain2(out, w)

        if steps == 0:
            return self.rgb(x)
        
        for step in range(steps):
            upscaled_out = F.interpolate(out, scale_factor=2, mode="bilinear")
            out = self.progressive_blocks[step](upscaled_out, w)

        upscaled_out = self.rgb_layers[steps-1](upscaled_out)
        out = self.progressive_blocks[steps](out)
        out = self.fade(alpha, upscaled_out, out)

        return out
    
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvolutionLayer(in_channels, out_channels)
        self.conv2 = ConvolutionLayer(out_channels, out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        return x
    
class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super().__init__()
        self.progressive_blocks = nn.ModuleList([])
        self.rgb_layers = nn.ModuleList([])
        self.leakyrelu  = nn.LeakyReLU(0.2, inplace=True)

        for i in range(len(factors) - 1, 0, -1):
            conv_in_channels = int(in_channels*factors[i])
            conv_out_channels = int(in_channels*factors[i-1])
            self.progressive_blocks.append(ConvolutionBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(ConvolutionLayer(img_channels, conv_in_channels, 1, 1, 0))

        self.rgb = ConvolutionLayer(img_channels, in_channels, 1, 1, 0)
        self.rgb_layers.append(self.rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.final_block = nn.Sequential(
            ConvolutionLayer(in_channels+1, in_channels),
            nn.LeakyReLU(0.2),
            ConvolutionLayer(in_channels, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            ConvolutionLayer(in_channels, 1, 1, 1, 0)
        )

    def fade(self, alpha, downscaled, out):
        return alpha*out + (1-alpha)*downscaled
    
    def batch_statistics(self, x):
        batch_stats = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_stats], dim=1)
    
    def forward(self, x, alpha, steps):
        current_step = len(self.progressive_blocks) - steps
        out = self.leakyrelu(self.rgb_layers[current_step](x))

        if steps == 0:
            out = self.batch_statistics(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        downscaled = self.leakyrelu(self.rgb_layers[current_step+1](self.avg_pool(x)))
        out = self.avg_pool(self.progressive_blocks[current_step](out))
        out = self.fade(alpha, downscaled, out)

        for step in range(current_step+1, len(self.progressive_blocks)):
            out = self.progressive_blocks[step](out)
            out = self.avg_pool(out)

        out = self.batch_statistics(out)
        return self.final_block(out).view(out.shape[0], -1)