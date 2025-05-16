import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import time

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer: Applies FFT, linear transform, and inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in x-direction
        self.modes2 = modes2  # Number of Fourier modes in y-direction
        self.modes3 = modes3  # Number of Fourier modes in t-direction

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        """
        Complex multiplication in Fourier space
        (batch, in_channel, x, y, t) × (in_channel, out_channel, x, y, t) → (batch, out_channel, x, y, t)
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier transform
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Apply learnable weights in Fourier space
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Inverse Fourier transform
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        3D Fourier Neural Operator:
        - Input: (batchsize, x, y, t, channels)
        - Output: (batchsize, x, y, t, output_channels)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 1  # Padding for non-periodic domains

        self.fc0 = nn.Linear(self.width, self.width)  # Lift input to hidden dimension

        # Spectral convolution layers
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)

        # Pointwise convolution layers
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        # Batch Normalization layers
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        # Fully connected layers to project to output
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, self.width)  # 4 output channels

    def forward(self, x):

        x = self.fc0(x)  # Lift to high-dimensional representation
        x = x.permute(0, 4, 1, 2, 3)  # Reshape for 3D convolution
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        # Spectral and local convolution layers
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Remove padding
        x = x[..., :-self.padding, :-self.padding, :-self.padding]

        x = x.permute(0, 2, 3, 4, 1)  # Reshape back
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        """
        Create a normalized coordinate grid.
        """
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float).reshape(1, size_x, 1, 1, 1).repeat(batchsize, 1, size_y, size_z, 1)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float).reshape(1, 1, size_y, 1, 1).repeat(batchsize, size_x, 1, size_z, 1)
        gridz = torch.linspace(0, 1, size_z, dtype=torch.float).reshape(1, 1, 1, size_z, 1).repeat(batchsize, size_x, size_y, 1, 1)
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)