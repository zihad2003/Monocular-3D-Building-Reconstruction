"""Shared-encoder / dual-decoder multi-task network."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead


class MultiTaskBuildingNet(nn.Module):
    """
    Shared encoder feeding two independent U-Net decoders:
      Head 1 — footprint logits
      Head 2 — non-negative height map (softplus)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        decoder_channels=(256, 128, 64, 32, 16),
    ):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=3, depth=5, weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels

        self.seg_decoder = UnetDecoder(
            encoder_channels=enc_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            attention_type=None,
        )
        self.height_decoder = UnetDecoder(
            encoder_channels=enc_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            attention_type=None,
        )

        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
            activation=None,
        )
        self.height_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            kernel_size=3,
            activation=None,
        )

    def forward(self, x):
        features = self.encoder(x)
        seg_feat = self.seg_decoder(*features)
        height_feat = self.height_decoder(*features)

        mask_logits = self.seg_head(seg_feat)
        height_map = F.softplus(self.height_head(height_feat))
        return mask_logits, height_map
