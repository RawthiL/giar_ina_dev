from __future__ import annotations

import timm
import torch
import torch.nn as nn

from allium_cepa_classifier.config.experiment_config import HeadConfig, ModelConfig


class BackboneWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def _build_head(cfg: HeadConfig, in_features: int, num_classes: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [in_features] + cfg.hidden_dims + [num_classes]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if cfg.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            elif cfg.activation == "relu":
                layers.append(nn.ReLU())
            elif cfg.activation == "gelu":
                layers.append(nn.GELU())
            if i < len(cfg.dropouts) and cfg.dropouts[i] > 0:
                layers.append(nn.Dropout(cfg.dropouts[i]))
    return nn.Sequential(*layers)


def freeze_model_stages(model: BackboneWithHead, arch: str, n: int) -> None:
    """Freeze all backbone stages except the last n. n=0 freezes everything."""
    if n == 0:
        return
    backbone = model.backbone

    if arch.startswith("efficientnet"):
        stages = [
            [backbone.conv_stem, backbone.bn1],
            [backbone.blocks[0]],
            [backbone.blocks[1]],
            [backbone.blocks[2]],
            [backbone.blocks[3]],
            [backbone.blocks[4]],
            [backbone.blocks[5]],
            [backbone.blocks[6]],
        ]
    elif arch == "resnet50":
        stages = [
            [backbone.conv1, backbone.bn1],
            [backbone.layer1],
            [backbone.layer2],
            [backbone.layer3],
            [backbone.layer4],
        ]
    elif arch == "vgg19":
        boundaries = [5, 10, 19, 28, 37]
        stages = [
            list(backbone.features[boundaries[i - 1] if i > 0 else 0 : boundaries[i]])
            for i in range(len(boundaries))
        ]
    else:
        raise ValueError(f"Unsupported arch for freezing: {arch}")

    for group in stages[: max(0, len(stages) - n)]:
        for module in group:
            for param in module.parameters():
                param.requires_grad = False


def build_model(cfg: ModelConfig, num_classes: int = 2) -> BackboneWithHead:
    backbone = timm.create_model(cfg.arch, pretrained=cfg.pretrained, num_classes=0)
    with torch.no_grad():
        in_features = backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
    head = _build_head(cfg.head, in_features, num_classes)
    model = BackboneWithHead(backbone, head)
    freeze_model_stages(model, cfg.arch, cfg.freeze_stages)
    return model
