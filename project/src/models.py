# """Generator, discriminator, critic, and classifier model definitions."""

# """Classifier model definitions."""
# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights


# def build_classifier(num_classes: int = 7, pretrained: bool = True):
#     """Build ResNet-18 and fine-tune all layers."""
#     weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#     model = resnet18(weights=weights)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     # Do not freeze any layers: all parameters are fine-tuned.
#     for parameter in model.parameters():
#         parameter.requires_grad = True
#     return model

"""Classifier and conditional GAN model definitions for HAM10000.

The existing ResNet-18 classifier API is preserved. GAN images use RGB tensors
in [-1, 1] and integer labels in [0, num_classes - 1].
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_classifier(num_classes: int = 7, pretrained: bool = True):
    """Build ResNet-18 and fine-tune all layers."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def _init_gan_weights(module: nn.Module, init_std: float = 0.02) -> None:
    """DCGAN-style initialization for convolution, linear, and BatchNorm layers."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, mean=1.0, std=init_std)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)


class ConditionalGenerator(nn.Module):
    """Class-conditional 64x64 RGB generator.

    Concatenates z (z_dim) with a learned class embedding, projects to 4x4xC,
    and upsamples 4 -> 8 -> 16 -> 32 -> 64.
    """
    def __init__(
        self,
        num_classes: int = 7,
        z_dim: int = 100,
        class_embed_dim: int = 50,
        base_channels: int = 512,
        image_channels: int = 3,
        init_std: float = 0.02,
    ):
        super().__init__()
        if base_channels < 64:
            raise ValueError("base_channels must be at least 64.")
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.project = nn.Linear(z_dim + class_embed_dim, base_channels * 4 * 4)
        self.net = nn.Sequential(
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels // 8, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.base_channels = base_channels
        self.apply(lambda module: _init_gan_weights(module, init_std))

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.z_dim:
            raise ValueError(f"noise must have shape (batch, {self.z_dim}).")
        labels = labels.long()
        conditioned = torch.cat((noise, self.class_embedding(labels)), dim=1)
        x = self.project(conditioned).view(-1, self.base_channels, 4, 4)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    """Conditional DCGAN discriminator returning one logit per image."""
    def __init__(
        self,
        num_classes: int = 7,
        class_embed_dim: int = 50,
        base_channels: int = 64,
        image_channels: int = 3,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.label_project = nn.Linear(class_embed_dim, 64 * 64)
        self.features = nn.Sequential(
            nn.Conv2d(image_channels + 1, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=True)
        self.apply(lambda module: _init_gan_weights(module, init_std))

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        label_map = self.label_project(self.label_embedding(labels)).view(-1, 1, 64, 64)
        x = torch.cat((images, label_map), dim=1)
        return self.output(self.features(x)).view(-1)


class ConditionalCritic(nn.Module):
    """Conditional WGAN-GP critic; returns unrestricted scalar scores.

    GroupNorm(1, C) is used when configured as layernorm, avoiding BatchNorm
    because gradient penalty is computed per input image.
    """
    def __init__(
        self,
        num_classes: int = 7,
        class_embed_dim: int = 50,
        base_channels: int = 64,
        image_channels: int = 3,
        norm: str = "layernorm",
        init_std: float = 0.02,
    ):
        super().__init__()
        if norm not in {"layernorm", "instancenorm", "none"}:
            raise ValueError("critic norm must be layernorm, instancenorm, or none.")
        self.label_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.label_project = nn.Linear(class_embed_dim, 64 * 64)

        layers = []
        channels = [image_channels + 1, base_channels, base_channels * 2,
                    base_channels * 4, base_channels * 8]
        for idx in range(4):
            layers.append(nn.Conv2d(channels[idx], channels[idx + 1], 4, 2, 1, bias=True))
            if idx > 0 and norm != "none":
                if norm == "layernorm":
                    layers.append(nn.GroupNorm(1, channels[idx + 1]))
                else:
                    layers.append(nn.InstanceNorm2d(channels[idx + 1], affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.features = nn.Sequential(*layers)
        self.output = nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=True)
        self.apply(lambda module: _init_gan_weights(module, init_std))

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_project(self.label_embedding(labels.long())).view(-1, 1, 64, 64)
        x = torch.cat((images, label_map), dim=1)
        return self.output(self.features(x)).view(-1)
