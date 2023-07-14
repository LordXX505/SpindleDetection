from functools import partial

import torch
import torch.nn as nn
import sys
from typing import Tuple, Any, List, Optional, Callable, Union
from utils.utils import UnetConvNd, UnetGridGatingSignalNd, GridAttentionBlock, UnetUp_1d, UnetDsv1
from utils.other import weights_init_normal, weights_init_xavier, weights_init_kaiming, weights_init_orthogonal


class SpindleUnet(torch.nn.Module):
    """
        def __init__(self,
                 feature_scale: int = 2,
                 n_classes: int = 1,
                 deconv: bool = True,
                 in_channels: int = 1,
                 nd: int = 1, convolution dimension
                 convs: int = 3, the number of convolution in one unet layer
                 kernel_size: Union[int, tuple[int]] = 11, kernel size default is 11
                 attention_dsample: int = 2, attention down sampler rate
                 is_batchnorm: bool = True,
                 Using_deep: bool = True, deep predict
                 drop_out: float = 0.2,
                 init_type: Optional[str] = None) -> None
        """
    def __init__(self, feature_scale: int = 2, n_classes: int = 1, deconv: bool = True, in_channels: int = 1, nd=1,
                 convs: int = 3, kernel_size: Union[int, Tuple[int]] = 11, attention_dsample: int = 2,
                 is_batchnorm: bool = True, Using_deep: bool = True, drop_out: float = 0.2, init_type: str = None
                 ) -> None:
        super(SpindleUnet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes
        if init_type is None:
            init_type = 'kaiming'
        self.init_type = init_type
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.conv1 = UnetConvNd(convs=convs, filter_size=kernel_size, padding=(kernel_size - 1) // 2, nd=nd,
                                batchnorm=is_batchnorm,
                                in_size=self.in_channels, out_size=filters[0], bias=True)
        if drop_out > 0:
            self.maxpool1 = nn.Sequential(nn.Dropout1d(0.2), nn.MaxPool1d(2))
            self.maxpool2 = nn.Sequential(nn.Dropout1d(0.2), nn.MaxPool1d(2))
            self.maxpool3 = nn.Sequential(nn.Dropout1d(0.2), nn.MaxPool1d(2))
            self.maxpool4 = nn.Sequential(nn.Dropout1d(0.2), nn.MaxPool1d(2))
        else:
            self.maxpool1 = nn.MaxPool1d(2)
            self.maxpool2 = nn.MaxPool1d(2)
            self.maxpool3 = nn.MaxPool1d(2)
            self.maxpool4 = nn.MaxPool1d(2)

        self.conv2 = UnetConvNd(convs=convs, filter_size=kernel_size, padding=(kernel_size - 1) // 2, nd=nd,
                                batchnorm=is_batchnorm, in_size=filters[0], out_size=filters[1], bias=True)

        self.conv3 = UnetConvNd(convs=convs, filter_size=kernel_size, padding=(kernel_size - 1) // 2, nd=nd,
                                batchnorm=is_batchnorm, in_size=filters[1], out_size=filters[2], bias=True)

        self.conv4 = UnetConvNd(convs=convs, filter_size=kernel_size, padding=(kernel_size - 1) // 2, nd=nd,
                                batchnorm=is_batchnorm, in_size=filters[2], out_size=filters[3], bias=True)

        self.conv5 = UnetConvNd(convs=convs, filter_size=kernel_size, padding=(kernel_size - 1) // 2, nd=nd,
                                batchnorm=is_batchnorm, in_size=filters[3], out_size=filters[4], bias=True)

        self.gating = UnetGridGatingSignalNd(filters[4], filters[4], kernel_size=1, is_batchnorm=is_batchnorm)

        # attention blocks
        self.attn1 = GridAttentionBlock(in_channels=filters[3], gating_channels=filters[4], inter_channels=filters[3],
                                        sub_sample_factor=attention_dsample)

        self.attn2 = GridAttentionBlock(in_channels=filters[2], gating_channels=filters[3], inter_channels=filters[2],
                                        sub_sample_factor=attention_dsample)

        self.attn3 = GridAttentionBlock(in_channels=filters[1], gating_channels=filters[2], inter_channels=filters[1],
                                        sub_sample_factor=attention_dsample)

        # up_blocks
        self.upconv1 = UnetUp_1d(convs=convs, filter_size=kernel_size, padding=(kernel_size - 1) // 2, nd=1,
                                 batchnorm=True,
                                 in_size=filters[4], out_size=filters[3], deconv=deconv, drop_out=drop_out)

        self.upconv2 = UnetUp_1d(convs=convs, filter_size=kernel_size, padding=(kernel_size - 1) // 2, nd=1,
                                 batchnorm=True, in_size=filters[3], out_size=filters[2], deconv=deconv,
                                 drop_out=drop_out)

        self.upconv3 = UnetUp_1d(convs=convs, filter_size=kernel_size, padding=(kernel_size - 1) // 2, nd=1,
                                 batchnorm=True,
                                 in_size=filters[2], out_size=filters[1], deconv=deconv, drop_out=drop_out)

        self.upconv4 = UnetUp_1d(convs=convs, filter_size=kernel_size, padding=(kernel_size - 1) // 2, nd=1,
                                 batchnorm=True, in_size=filters[1], out_size=filters[0], deconv=deconv,
                                 drop_out=drop_out)

        # deep supervision
        self.Using_deep = Using_deep
        if Using_deep:
            self.dsv1 = UnetDsv1(in_size=filters[3], out_size=n_classes, scale_factor=8)
            self.dsv2 = UnetDsv1(in_size=filters[2], out_size=n_classes, scale_factor=4)
            self.dsv3 = UnetDsv1(in_size=filters[1], out_size=n_classes, scale_factor=2)
            self.dsv4 = nn.Conv1d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)
            # final conv (without any concat)
            self.final = nn.Conv1d(n_classes * 4, n_classes, kernel_size=1)
        else:
            self.final = nn.Conv1d(filters[0], n_classes, kernel_size=1)

        if n_classes == 1:
            self.final = nn.Conv1d(filters[0], 2, kernel_size=1)

        self.soft = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        print('initialization method [%s]' % self.init_type)
        if self.init_type == 'normal':
            self.apply(weights_init_normal)
        elif self.init_type == 'xavier':
            self.apply(weights_init_xavier)
        elif self.init_type == 'kaiming':
            self.apply(weights_init_kaiming)
        elif self.init_type == 'orthogonal':

            self.apply(weights_init_orthogonal)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        # print("conv1, maxpool1", maxpool1.shape)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # print("conv2, maxpool2", maxpool2.shape)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # print("conv3, maxpool3", maxpool3.shape)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        # print("conv4, maxpool4", maxpool4.shape)

        # Gating Signal Generation
        center = self.conv5(maxpool4)
        gating = self.gating(center)
        # print('center, gating', center.shape, gating.shape)
        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att1 = self.attn1(conv4, gating)
        # print('attn1', g_conv4.shape)
        up1 = self.upconv1(g_conv4, center)
        # print('upconv1', up1.shape)
        g_conv3, att2 = self.attn2(conv3, up1)
        # print('attn2', g_conv3.shape)
        up2 = self.upconv2(g_conv3, up1)
        # print('upconv2', up2.shape)
        g_conv2, att3 = self.attn3(conv2, up2)
        # print("att3", g_conv4.shape)
        up3 = self.upconv3(g_conv2, up2)
        # print("upconv3", up3.shape)

        up4 = self.upconv4(conv1, up3)
        # print("upconv4", up4.shape)
        # Deep Supervision
        if self.n_classes == 1:
            final = self.final(up4)
            final = self.soft(final)
            pred = final[:, 1, :]
            pred = pred.squeeze(1)
        else:
            if self.Using_deep:
                dsv4 = self.dsv4(up4)
                dsv3 = self.dsv3(up3)
                dsv2 = self.dsv2(up2)
                dsv1 = self.dsv1(up1)
                final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
            else:
                final = self.final(up4)
            pred = final
        return pred



def Unet_drop_fs1(**kwargs):
    """
    def __init__(self,
             feature_scale: int = 2, define
             n_classes: int = 1, define
             deconv: bool = True,
             in_channels: int = 1, define
             nd: int = 1, define
             convs: int = 3,
             kernel_size: Union[int, tuple[int]] = 11,
             attention_dsample: int = 2, define
             is_batchnorm: bool = True,
             Using_deep: bool = True,
             drop_out: float = 0.2,
             init_type: Optional[str] = None) -> None
    """
    model = SpindleUnet(
        feature_scale=1, in_channels=1, nd=1, attention_dsample=2, n_classes=2, **kwargs
    )
    return model


def Unet_drop_fs2(**kwargs):
    model = SpindleUnet(
        feature_scale=2, in_channels=1, nd=1, attention_dsample=2, n_classes=2, **kwargs
    )
    return model


def Unet_drop_fs2_fnfp(**kwargs):
    model = SpindleUnet(
        feature_scale=2, in_channels=1, nd=1, attention_dsample=2, n_classes=1, **kwargs
    )
    return model


def Unet_drop_fs4(**kwargs):
    model = SpindleUnet(
        feature_scale=4, in_channels=1, nd=1, attention_dsample=2, n_classes=2, **kwargs
    )
    return model

