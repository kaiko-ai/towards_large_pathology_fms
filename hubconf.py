"""Pathology FM model hub from kaiko.ai."""

import timm
from timm.models import vision_transformer
import torch
from torch import nn

dependencies = ["torch", "timm"]
# List of package names required to load the model


RELEASE_TAG = "0.0.1"
"""The release tag to fetch the weights from."""


def vits16(*, pretrained: bool = True, dynamic_img_size: bool = True) -> nn.Module:
    """Initializes the vision transformer ViTS-16 pathology FM by kaiko.ai.

    Args:
        pretrained: Whether to load the pretrained model weights.
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.

    Returns:
        The torch ViTS-16 based foundation model.
    """
    model = timm.create_model(
        model_name="vit_small_patch16_224",
        dynamic_img_size=dynamic_img_size,
        num_classes=0,
        pretrained=False,
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vits16.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)

    return model


def vits8(*, pretrained: bool = True, dynamic_img_size: bool = True) -> nn.Module:
    """Initializes the vision transformer ViTS-8 pathology FM by kaiko.ai.

    Args:
        pretrained: Whether to load the pretrained model weights.
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.

    Returns:
        The torch ViTS-8 based foundation model.
    """
    model = timm.create_model(
        model_name="vit_small_patch8_224",
        dynamic_img_size=dynamic_img_size,
        num_classes=0,
        pretrained=False,
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vits8.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)

    return model


def vitb16(*, pretrained: bool = True, dynamic_img_size: bool = True) -> nn.Module:
    """Initializes the vision transformer ViTB-16 pathology FM by kaiko.ai.

    Args:
        pretrained: Whether to load the pretrained model weights.
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.

    Returns:
        The torch ViTB-16 based foundation model.
    """
    model = timm.create_model(
        model_name="vit_base_patch16_224",
        dynamic_img_size=dynamic_img_size,
        num_classes=0,
        pretrained=False,
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vitb16.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)

    return model


def vitb8(*, pretrained: bool = True, dynamic_img_size: bool = True) -> nn.Module:
    """Initializes the vision transformer ViTB-8 pathology FM by kaiko.ai.

    Args:
        pretrained: Whether to load the pretrained model weights.
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.

    Returns:
        The torch ViTB-8 based foundation model.
    """
    model = timm.create_model(
        model_name="vit_base_patch8_224",
        dynamic_img_size=dynamic_img_size,
        num_classes=0,
        pretrained=False,
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vitb8.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)

    return model


def vitl14(*, pretrained: bool = True, dynamic_img_size: bool = True) -> nn.Module:
    """Initializes the vision transformer ViTL-14 pathology FM by kaiko.ai.

    Args:
        pretrained: Whether to load the pretrained model weights.
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.

    Returns:
        The torch ViTL-14 based foundation model.
    """
    model = timm.create_model(
        model_name="vit_large_patch14_reg4_dinov2",
        dynamic_img_size=dynamic_img_size,
        pretrained=False,
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vitl14.pth",
            map_location="cpu",
        )
        adapted_state_dict = vision_transformer._convert_dinov2(state_dict, model)
        model.load_state_dict(adapted_state_dict, strict=True)

    return model
