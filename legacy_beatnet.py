"""Legacy BEATNet definition matching the saved attention decoder checkpoint.

Architecture and settings preserved from breat_segmentation_ui/backend/worker.py.
"""
import torch
import torch.nn as nn
from monai.networks.nets import FlexibleUNet

class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_excitation = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        chn_se = self.channel_excitation(x) * x
        spa_se = self.spatial_excitation(x) * x
        return chn_se + spa_se

class AttentionAllDecoderUNet(FlexibleUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        decoder_channels = kwargs.get("decoder_channels", (512, 256, 128, 64, 32))
        self.attention_blocks = nn.ModuleList([
            SCSEBlock(ch) for ch in decoder_channels
        ])

    def forward(self, x):
        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        x = features[-1]

        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            x = self.attention_blocks[i](x)

        x = self.segmentation_head(x)
        return x
