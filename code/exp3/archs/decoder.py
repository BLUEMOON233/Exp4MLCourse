# -*- coding: utf-8 -*-
"""
改进版特征解码器 (Feature Decoder)
使用 Pixel Shuffle 上采样替代双线性插值，减少模糊
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    标准的残差块：Conv -> BN -> ReLU -> Conv -> BN -> Sum
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class PixelShuffleUpsample(nn.Module):
    """
    Pixel Shuffle 上采样模块
    比双线性插值保留更多高频细节，减少模糊
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(PixelShuffleUpsample, self).__init__()
        # Pixel Shuffle 需要 out_channels * scale^2 的中间通道
        mid_channels = out_channels * (scale_factor ** 2)
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    """
    通道注意力模块 (SE Block 风格)
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlockWithAttention(nn.Module):
    """
    带通道注意力的残差块
    """
    def __init__(self, channels):
        super(ResBlockWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)
        out += residual
        out = self.relu(out)
        return out


class FeatureDecoder(nn.Module):
    """
    改进版解码器：使用 Pixel Shuffle 上采样
    
    输入: VGG Block1 特征 [B, 128, 56, 56]
    输出: 重建图像 [B, 3, 224, 224]
    
    改进点:
    1. 使用 Pixel Shuffle 替代双线性插值，减少模糊
    2. 增加通道注意力机制
    3. 更深的残差网络
    """
    def __init__(self, in_channels=128, out_channels=3, num_res_blocks=4):
        super(FeatureDecoder, self).__init__()
        
        # 1. 初始特征变换
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 2. 深层残差处理 (在 56×56 分辨率)
        # 使用带注意力的残差块
        res_blocks = []
        for i in range(num_res_blocks):
            if i % 2 == 0:
                res_blocks.append(ResBlockWithAttention(256))
            else:
                res_blocks.append(ResBlock(256))
        self.res_body = nn.Sequential(*res_blocks)
        
        # 3. 上采样 Stage 1: 56 -> 112 (使用 Pixel Shuffle)
        self.up1 = nn.Sequential(
            PixelShuffleUpsample(256, 128, scale_factor=2),
            ResBlock(128)
        )
        
        # 4. 上采样 Stage 2: 112 -> 224 (使用 Pixel Shuffle)
        self.up2 = nn.Sequential(
            PixelShuffleUpsample(128, 64, scale_factor=2),
            ResBlock(64)
        )
        
        # 5. 输出层
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入: [B, 128, 56, 56]
        x = self.initial_conv(x)    # [B, 256, 56, 56]
        x = self.res_body(x)        # [B, 256, 56, 56]
        x = self.up1(x)             # [B, 128, 112, 112]
        x = self.up2(x)             # [B, 64, 224, 224]
        x = self.output(x)          # [B, 3, 224, 224]
        return x


class FeatureDecoderLegacy(nn.Module):
    """
    原版解码器 (使用双线性插值，保留用于对比)
    """
    def __init__(self, in_channels=128, out_channels=3, num_res_blocks=3):
        super(FeatureDecoderLegacy, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(128))
        self.res_body = nn.Sequential(*res_blocks)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64)
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_body(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return x
