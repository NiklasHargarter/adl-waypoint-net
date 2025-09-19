# resnet_features_backbones.py
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


from transformers import ResNetModel

from external.Conv_Adapter.models.backbones import resnet as conv_adapter_resnet


from peft import LoraConfig, get_peft_model


@dataclass
class BackboneBuildConfig:
    variant: str = "normal"  # "normal" | "lora" | "dora" | "convadapter"

    # LoRA/DoRA (simplified to most common settings)
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    use_dora: bool = False  # Whether to use DoRA (True) or LoRA (False)

    # Conv-Adapter
    adapter_dim: int = 64  # bottleneck width inside adapters
    stages: str = "late"  # "all" | "late"
    adapt_size: int = 8  # for grouping in ConvAdapter


class NormalResNetFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        feats = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.features = feats
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.features(x)
        return output.last_hidden_state


class LoRADoRAResNetFeatures(nn.Module):
    """
    Wrap ResNet-50 features with LoRA or DoRA adapters on Conv2d/Linear modules.
    """

    def __init__(self, cfg: BackboneBuildConfig):
        super().__init__()
        feats = ResNetModel.from_pretrained("microsoft/resnet-50")

        # Ensure parameters are contiguous to avoid issues with PEFT LoRA/DoRA. No idea copilot recommended it.
        for param in feats.parameters():
            param.data = param.data.contiguous()

        # Choose target modules: conv1, conv2, conv3, and downsample from block3 + block4
        names = [
            n for n, m in feats.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))
        ]
        target_modules = [
            n
            for n in names
            if (
                ("encoder.stages.2" in n or "encoder.stages.3" in n)
                and (
                    ".layer.0.convolution" in n  # conv1
                    or ".layer.1.convolution" in n  # conv2
                    or ".layer.2.convolution" in n  # conv3
                    or ".shortcut.convolution" in n  # downsample
                )
            )
        ]
        if len(target_modules) == 0:
            raise RuntimeError(
                "No target modules found for LoRA/DoRA on ResNet-50 features"
            )

        lcfg = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            bias=cfg.bias,
            target_modules=target_modules,
            use_dora=cfg.use_dora,  # Dora toggle
        )
        peft_feats = get_peft_model(feats, lcfg)
        # Ensure PEFT parameters are contiguous to avoid LoRA/DoRA issues
        for param in peft_feats.parameters():
            if param.requires_grad:
                param.data = param.data.contiguous()

        self.features = peft_feats
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.features(x)
        return output.last_hidden_state

    def merge_and_unload(self):
        if hasattr(self.features, "merge_and_unload"):
            self.features = self.features.merge_and_unload()
            return self.features
        raise AttributeError("merge_and_unload not available")


class ConvAdapterResNetFeatures(nn.Module):
    """
    ResNet-50 features with Conv-Adapter using the reference implementation.
    """

    def __init__(self, cfg: BackboneBuildConfig):
        super().__init__()
        # Use reference ResNet with ConvAdapter integrated
        tuning_config = {"method": "conv_adapt", "adapt_size": cfg.adapt_size}
        ref_model = conv_adapter_resnet.resnet50(
            pretrained=True, tuning_config=tuning_config
        )

        # Dynamically slice the model to stop after layer4 (spatial features)
        # ref_model children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, norm, avgpool, head]
        # Take up to layer4 (indices 0-8)
        self.features = nn.Sequential(*list(ref_model.children())[:9])

        # Freeze base parameters (everything except tuning modules)
        for name, param in ref_model.named_parameters():
            if "tuning_module" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def build_resnet_backbone(bcfg: BackboneBuildConfig) -> Tuple[nn.Module, int]:
    if bcfg.variant == "normal":
        m = NormalResNetFeatures()
        return m, m.out_channels
    if bcfg.variant in ("lora", "dora"):
        # Set use_dora based on variant
        bcfg.use_dora = bcfg.variant == "dora"
        m = LoRADoRAResNetFeatures(bcfg)
        return m, m.out_channels
    if bcfg.variant == "convadapter":
        m = ConvAdapterResNetFeatures(bcfg)
        return m, m.out_channels
    raise ValueError(f"Unknown variant: {bcfg.variant}")
