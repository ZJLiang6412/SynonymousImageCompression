import torch
from torch import nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type='A', **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        # type 'A' to mask the center, from PixelCNN
        if mask_type not in ('A', 'B'):
            raise ValueError("Invalid \"mask_type\" value: {}".format(mask_type))
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, h, w = self.mask.size()
        # setting below weights to 0
        self.mask[:, :, (h // 2), (w // 2 + (mask_type == 'B')):] = 0
        self.mask[:, :, (h // 2 + 1):, :] = 0

    def forward(self, input_):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(input_)


class MaskedConv2d_AR(nn.Conv2d):
    def __init__(self, *args, num_slices=16, **kwargs):
        super(MaskedConv2d_AR, self).__init__(*args, **kwargs)
        # type 'A' to mask the center, from PixelCNN
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.num_slices = num_slices
        Cout, Cin, h, w = self.mask.size()
        Cout_semSection, Cin_semSection = Cout // num_slices, Cin // num_slices
        for c_o in range(num_slices):
            for c_i in range(num_slices):
                if c_o == c_i:
                    # setting left-up weights (with center) to 0
                    self.mask[c_o * Cout_semSection:(c_o + 1) * Cout_semSection, c_i * Cin_semSection:(c_i + 1) * Cin_semSection, (h // 2), (w // 2):] = 0
                    self.mask[c_o * Cout_semSection:(c_o + 1) * Cout_semSection, c_i * Cin_semSection:(c_i + 1) * Cin_semSection, (h // 2 + 1):, :] = 0
                elif c_o < c_i:
                    # setting all weights to 0
                    self.mask[c_o * Cout_semSection:(c_o + 1) * Cout_semSection, c_i * Cin_semSection:(c_i + 1) * Cin_semSection, :, :] = 0
                # else:   # c_o > c_i
                    # setting all weights (without center) to 1 (coding every layer, the former layer can be full reference for next layer)

    def forward(self, input_):
        self.weight.data *= self.mask
        return super(MaskedConv2d_AR, self).forward(input_)


class ContextModel(nn.Module):
    def __init__(self, num_channels=256, num_slices=16):
        super(ContextModel, self).__init__()
        self.num_channels = num_channels
        num_channels_out = self.num_channels * 2
        self.num_slices = num_slices

        self.layer = MaskedConv2d(num_slices=num_slices, in_channels=self.num_channels, out_channels=num_channels_out,
                  kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

    def forward(self, input_):
        out_ = self.layer(input_)
        return out_


class ContextModel_AR(nn.Module):
    def __init__(self, num_channels=256, num_slices=16):
        super(ContextModel_AR, self).__init__()
        self.num_channels = num_channels
        num_channels_out = self.num_channels * 2
        self.num_slices = num_slices

        self.layer = MaskedConv2d_AR(num_slices=num_slices, in_channels=self.num_channels, out_channels=num_channels_out,
                  kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

    def forward(self, input_):
        B, C, H, W = input_.shape
        out_ = self.layer(input_)
        out_ = out_.reshape(B, self.num_slices, 2, -1, H, W)
        out_ = out_.permute(0, 2, 1, 3, 4, 5).reshape(B, -1, H, W)
        return out_
