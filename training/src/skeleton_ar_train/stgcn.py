"""Spatial Temporal Graph Convolutional Network (ST-GCN).

Reference:
    Yan, Xiong, Lin, "Spatial Temporal Graph Convolutional Networks for
    Skeleton-Based Action Recognition", AAAI 2018.

This is a clean re-implementation of the published architecture using
plain PyTorch (no mmcv / mmaction dependency), targeting COCO-17
skeletons. The graph partitioning follows `graph.partition_adjacency`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Spatial graph convolution over a fixed `K`-partition adjacency.

    Input:  (B, C_in,  T, V)
    Output: (B, C_out, T, V)
    """

    def __init__(self, in_channels: int, out_channels: int, num_partitions: int) -> None:
        super().__init__()
        self.k = num_partitions
        self.conv = nn.Conv2d(in_channels, out_channels * num_partitions, kernel_size=1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        b, _, t, v = x.shape
        h = self.conv(x)  # (B, K * C_out, T, V)
        h = h.view(b, self.k, -1, t, v)
        # Einsum over partitions: sum_k A_k @ h_k
        return torch.einsum("nkctv,kvw->nctw", h, adjacency).contiguous()


class STGCNBlock(nn.Module):
    """Spatial graph conv -> BatchNorm -> ReLU -> temporal conv -> ReLU -> residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_partitions: int,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        padding = (temporal_kernel - 1) // 2
        self.spatial = GraphConv(in_channels, out_channels, num_partitions)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.temporal = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=(temporal_kernel, 1),
                stride=(stride, 1), padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual: nn.Module = nn.Identity()
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.relu(self.bn1(self.spatial(x, adjacency)))
        h = self.temporal(h)
        h = self.dropout(h)
        return self.relu(h + residual)


class STGCN(nn.Module):
    """End-to-end ST-GCN classifier.

    Input shape: (B, C, T, V, M) where M is the number of persons per
    clip. M is averaged out before the final classifier.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_keypoints: int,
        adjacency: torch.Tensor,
        hidden_channels: tuple[int, ...] = (64, 64, 128, 128, 256, 256),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if adjacency.dim() != 3:
            raise ValueError("adjacency must be (K, V, V)")
        self.register_buffer("adjacency", adjacency)
        self.input_bn = nn.BatchNorm1d(in_channels * num_keypoints)

        self.blocks = nn.ModuleList()
        prev = in_channels
        for i, c in enumerate(hidden_channels):
            stride = 2 if i in (2, 4) else 1
            self.blocks.append(
                STGCNBlock(
                    prev, c,
                    num_partitions=adjacency.shape[0],
                    stride=stride,
                    dropout=dropout if i >= 2 else 0.0,
                    residual=(i > 0),
                )
            )
            prev = c

        self.classifier = nn.Conv2d(prev, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, v, m = x.shape
        # (B, M, V, C, T) -> (B*M, C, T, V) for per-person processing.
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(b * m, v * c, t)
        x = self.input_bn(x).view(b, m, v, c, t).permute(0, 1, 3, 4, 2)
        x = x.reshape(b * m, c, t, v)

        for block in self.blocks:
            x = block(x, self.adjacency)

        # Global average over T and V; average across persons (M).
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(b, m, -1, 1, 1).mean(dim=1)
        return self.classifier(x).flatten(1)
