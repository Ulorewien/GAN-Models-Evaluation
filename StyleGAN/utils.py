import torch
from torch import nn

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
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_slope = LinearLayer(w_dim, channels)
        self.style_bias = LinearLayer(w_dim, channels)

    def forward(self, x, w):
        slope = self.style_slope(w).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        x = self.instance_norm(x)
        
        return (slope*x) + bias