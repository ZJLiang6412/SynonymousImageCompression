import torch
from torch import nn


class EntropyParameters(nn.Module):
    def __init__(self, num_channels=256*2*2, num_slices=16):
        super(EntropyParameters, self).__init__()
        self.num_channels = num_channels
        self.num_slices = num_slices

        self.layer = nn.Sequential(
                nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels * 3 // 4, kernel_size=(1, 1), stride=(1, 1)),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=self.num_channels * 3 // 4, out_channels=self.num_channels * 3 // 4, kernel_size=(1, 1), stride=(1, 1)),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=self.num_channels * 3 // 4, out_channels=self.num_channels * 1 // 2, kernel_size=(1, 1), stride=(1, 1))
            )

    def forward(self, input_):
        return self.layer(input_)


class GroupEntropyParameters(nn.Module):
    def __init__(self, num_channels=256*2*2, num_slices=16):
        super(GroupEntropyParameters, self).__init__()
        self.num_channels = num_channels
        self.num_slices = num_slices

        self.layer = nn.Sequential(
                nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels * 3 // 4, kernel_size=(1, 1), stride=(1, 1), groups=num_slices),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=self.num_channels * 3 // 4, out_channels=self.num_channels * 3 // 4, kernel_size=(1, 1), stride=(1, 1), groups=num_slices),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=self.num_channels * 3 // 4, out_channels=self.num_channels * 1 // 2, kernel_size=(1, 1), stride=(1, 1), groups=num_slices)
            )

    def forward(self, input_):
        B, C, H, W = input_.shape
        input_ = input_.reshape(B, 4, self.num_slices, C // 4 // self.num_slices, H, W)
        input_ = input_.permute(0, 2, 1, 3, 4, 5).reshape(B, -1, H, W)
        out_ = self.layer(input_)
        out_ = out_.reshape(B, self.num_slices, 2, -1, H, W)
        out_ = out_.permute(0, 2, 1, 3, 4, 5).reshape(B, -1, H, W)
        return out_