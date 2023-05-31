import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn.functional import linear, relu


class ISTANet(nn.Module):
    def __init__(self, ista_layers: int, filter_count: int, filter_size: int, cha_color: int):
        """ "ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing" - https://arxiv.org/abs/1706.07929

        Args:
            `ista_layers` (int): Number of "ISTA-Modules".
            `filter_count` (int): Maximum number of filters.
            `filter_size` (int): Size of each filter.
            `cha_color` (int, optional): Number of color channels.
        """
        super(ISTANet, self).__init__()

        self.T = ista_layers
        self.rho = nn.Parameter(torch.ones(self.T) * 0.5)
        self.theta = nn.Parameter(torch.ones(self.T) * 0.01)
        self.F1 = nn.ModuleList()
        self.F2 = nn.ModuleList()
        self.B1 = nn.ModuleList()
        self.B2 = nn.ModuleList()
        for _ in range(self.T):
            self.F1.append(nn.Conv2d(cha_color, filter_count, kernel_size=filter_size, padding="same", bias=False))
            self.F2.append(nn.Conv2d(filter_count, filter_count, kernel_size=filter_size, padding="same", bias=False))
            self.B1.append(nn.Conv2d(filter_count, filter_count, kernel_size=filter_size, padding="same", bias=False))
            self.B2.append(nn.Conv2d(filter_count, cha_color, kernel_size=filter_size, padding="same", bias=False))

        for i in range(self.T):
            xavier_normal_(self.F1[i].weight)
            xavier_normal_(self.F2[i].weight)
            xavier_normal_(self.B1[i].weight)
            xavier_normal_(self.B2[i].weight)

    def forward(self, Y: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Qinit: torch.Tensor):
        """
        Args:
            `Y` (torch.Tensor): Low-resolution input. (batch, channel, hight x width)
            `A` (torch.Tensor): Φ†Φ, where Φ is the downsampling matrix.
            `B` (torch.Tensor): Φ†Y, where Y is a batch of flattened low-resolution input images.
        """

        x = linear(Y, Qinit, bias=None)  # Performs downsampling on the entire batch

        dim_b, dim_c, dim_hw = x.shape
        dim_h = dim_w = int(dim_hw**0.5)

        error_symmetry = []
        for i in range(self.T):
            r = x - self.rho[i] * linear(x, A) + self.rho[i] * B
            r = r.reshape(shape=(dim_b, dim_c, dim_h, dim_w))
            q = self.F2[i](relu(self.F1[i](r)))  # F(.)
            x = torch.sign(q) * relu(torch.abs(q) - self.theta[i])  # softshrink
            x = self.B2[i](relu(self.B1[i](x)))  # F^-1(.)
            r_rcvrd = self.B2[i](relu(self.B1[i](q)))  # F^-1(F(.))
            x = x.reshape(dim_b, dim_c, dim_hw)

            error_symmetry.append(r_rcvrd - r)

        return x, error_symmetry
