# resnet_gru_waypoint_net.py
import torch
import torch.nn as nn

from external.transfuser.team_code_transfuser.config import GlobalConfig
from .small_cnn import SmallCNN
from .resnet_features_backbones import BackboneBuildConfig, build_resnet_backbone


class MultimodalWaypointNet(nn.Module):
    """
    Multimodal waypoint predictor with a ResNet-50 RGB backbone:
      - variant = "normal" | "lora" | "dora" | "convadapter"
      - returns k future (x, y) offsets
    """

    def __init__(
        self,
        config: GlobalConfig | None = None,
        enc_dim_lidar: int = 128,
        hidden: int = 256,
        backbone_variant: str = "normal",  # "normal" | "lora" | "dora" | "convadapter"
        adapter_dim: int = 64,
        adapter_stages: str = "late",  # "late" | "all"
    ):
        super().__init__()
        if config is None:
            config = GlobalConfig()
        self.k_wp = int(config.pred_len)

        # Build backbone
        bcfg = BackboneBuildConfig(
            variant=backbone_variant,
            adapter_dim=adapter_dim,
            stages=adapter_stages,
        )
        rgb_features, rgb_out_ch = build_resnet_backbone(bcfg)
        self.rgb_features = rgb_features
        self.rgb_out_ch = rgb_out_ch

        # GAP to get a vector from spatial map
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # LiDAR encoder and fusion + GRU head
        self.lidar_enc = SmallCNN(in_ch=2, out_ch=enc_dim_lidar)
        fused_in = self.rgb_out_ch + enc_dim_lidar
        self.fuse = nn.Sequential(nn.Linear(fused_in, hidden), nn.ReLU(inplace=True))
        self.gru = nn.GRUCell(4, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(
        self, rgb: torch.Tensor, lidar_bev: torch.Tensor, target_point: torch.Tensor
    ) -> torch.Tensor:
        f_rgb = self.rgb_features(rgb)  # (B, C, H, W)
        f_rgb = self.gap(f_rgb).flatten(1)  # (B, C)
        f_lidar = self.lidar_enc(lidar_bev)
        z = self.fuse(torch.cat([f_rgb, f_lidar], dim=1))

        B = rgb.size(0)
        x = rgb.new_zeros((B, 2))
        outs = []
        for _ in range(self.k_wp):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.gru(x_in, z)
            dx = self.head(z)
            x = x + dx
            outs.append(x)
        return torch.stack(outs, dim=1)
