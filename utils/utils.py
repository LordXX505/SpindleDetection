# MIT License
#
# Copyright (c) 2018 Ozan Oktay


import numpy as np
import torch
from torch import nn
from typing import *
from .other import init_weights
import torch.nn.functional as F


class GridAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int = None,
                 sub_sample_factor: Union[List, Tuple, int] = 2, mode: str = 'concatenation', dimension: int = 1):
        super(GridAttentionBlock, self).__init__()
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = sub_sample_factor
        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
            self.upsample_mode = 'linear'
        else:
            raise NotImplemented

        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Initialise weights
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.shape
        batch_size = input_size[0]

        assert batch_size == g.shape[0]

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class UnetConvNd(nn.Module):
    def __init__(self, convs: int = 2,  filter_size: Union[int, Tuple[int]] = None,
                 padding: Union[str, int, Tuple[int]] = 0, stride: Union[int, Tuple[int]] = 1,
                 nd: int = 1, batchnorm: bool = True,
                 in_size: int = None, out_size: int = None, bias: bool = True, init_type: str ='kaiming'
                 ) -> None:
        super(UnetConvNd, self).__init__()
        assert in_size is not None
        assert out_size is not None
        if filter_size is None:
            filter_size = tuple([1 for i in range(nd)])
        self.init_type = init_type
        self.blocks = nn.ModuleList()
        self.blocks.append(block(filter_size, padding, stride, nd, batchnorm, in_size, out_size, bias, init_type))
        for i in range(convs-1):
            self.blocks.append(block(filter_size, padding, stride, nd, batchnorm, out_size, out_size, bias, init_type))

        # # initialize weight
        # for _, m in self.named_modules():
        #     print(_)
        #     init_weights(m, init_type)

    def init_weight(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        init_weights(m,  self.init_type)

    def forward(self, x):
        for _block in self.blocks:
            x = _block(x)
        return x


class block(nn.Module):
    def __init__(self, filter_size: Union[int, Tuple[int]] = None,
                 padding: Union[str, int, Tuple[int]] = 0, stride: Union[int, Tuple[int]] = 1,
                 nd: int = 1, batchnorm: bool = True,
                 in_size: int = None, out_size: int = None, bias: bool = True, init_type: str ='kaiming') -> None:
        super(block, self).__init__()
        if nd == 1:
            conv1 = torch.nn.Conv1d(in_size, out_size, filter_size, stride, padding, bias=bias)
            if batchnorm:
                batchnormd = torch.nn.BatchNorm1d(out_size)
        elif nd == 2:
            conv1 = torch.nn.Conv2d(in_size, out_size, filter_size, stride, padding, bias=bias)
            if batchnorm:
                batchnormd = torch.nn.BatchNorm2d(out_size)
        else:
            conv1 = torch.nn.Conv3d(in_size, out_size, filter_size, stride, padding, bias=bias)
            if batchnorm:
                batchnormd = torch.nn.BatchNorm2d(out_size)

        if batchnorm:
            self.block = nn.Sequential(conv1, batchnormd, nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(conv1,  nn.ReLU(inplace=True))

        # # initialize weight
        # for _, m in self.named_modules():
        #     print(_)
        #     init_weights(m, init_type)

    def forward(self, x):
        return self.block(x)


class UnetUp_1d(nn.Module):

    def __init__(self, convs: int = 2,  filter_size: Union[int, Tuple[int]] = None,
                 padding: Union[str, int, Tuple[int]] = 0, stride: Union[int, Tuple[int]] = 1,
                 nd: int = 1, batchnorm: bool = True,
                 in_size: int = None, out_size: int = None, bias: bool = True, init_type: str = 'kaiming',
                 scale_factor: Union[Tuple, int] = 2, deconv: bool = True, drop_out: float = 0) -> None:
        super(UnetUp_1d, self).__init__()
        self.drop_out = drop_out
        if deconv:
            self.conv = UnetConvNd(convs, filter_size, padding, stride, nd, batchnorm, in_size, out_size,
                                   bias, init_type)
            self.up = nn.ConvTranspose1d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.conv = UnetConvNd(convs, filter_size, padding, stride, nd, batchnorm, in_size + out_size, out_size,
                                   bias, init_type)
            self.up = F.interpolate(scale_factor=scale_factor, mode='linear')

        # # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('UnetConvNd') != -1:
        #         continue
        #     init_weights(m, init_type='kaiming')
        #     print(m.__class__.__name__)

    def forward(self, inputs1, inputs2):
        if self.drop_out > 0:
            Dropout = nn.Dropout(self.drop_out)
            inputs2 = Dropout(inputs2)
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [torch.div(offset, 2, rounding_mode='floor')]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetGridGatingSignalNd(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, is_batchnorm=True, nd=1) -> None:
        super(UnetGridGatingSignalNd, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv1d(in_size, out_size, kernel_size, 1, 0),
                                       nn.BatchNorm1d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv1d(in_size, out_size, kernel_size, 1, 0),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class UnetDsv1(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv1, self).__init__()
        self.scale_factor = scale_factor
        self.dsv = nn.Sequential(nn.Conv1d(in_size, out_size, kernel_size=1, stride=1, padding=0))

    def forward(self, input):
        input = self.dsv(input)
        return F.interpolate(input, scale_factor=self.scale_factor, mode='linear')
