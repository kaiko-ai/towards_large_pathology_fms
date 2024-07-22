"""Pathology FM model hub from kaiko.ai."""

import timm
import torch
from torch import nn

dependencies = ["torch", "timm"]
# List of package names required to load the model


RELEASE_TAG = "0.0.1"
"""The release tag to fetch the weights from."""


def vits16(dynamic_img_size: bool = True, out_indices: int | List[int] | None = None) -> nn.Module:
    """Initializes the vision transformer ViTS-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-16 based foundation model.
    """
    return timm.create_model(
        model_name="vit_small_patch16_224",
        dynamic_img_size=dynamic_img_size,
        pretrained_cfg={
            "url": f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vits16.pth",
            "num_classes": 0
        },
        pretrained=True,
        out_indices=out_indices,
        features_only=out_indices is not None,
    )


def vits8(dynamic_img_size: bool = True, out_indices: int | List[int] | None = None) -> nn.Module:
    """Initializes the vision transformer ViTS-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTS-8 based foundation model.
    """
    return timm.create_model(
        model_name="vit_small_patch8_224",
        dynamic_img_size=dynamic_img_size,
        pretrained_cfg={
            "url": f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vits8.pth",
            "num_classes": 0
        },
        pretrained=True,
        out_indices=out_indices,
        features_only=out_indices is not None,
    )


def vitb16(dynamic_img_size: bool = True, out_indices: int | List[int] | None = None) -> nn.Module:
    """Initializes the vision transformer ViTB-16 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTB-16 based foundation model.
    """
    return timm.create_model(
        model_name="vit_base_patch16_224",
        dynamic_img_size=dynamic_img_size,
        pretrained_cfg={
            "url": f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vitb16.pth",
            "num_classes": 0
        },
        pretrained=True,
        out_indices=out_indices,
        features_only=out_indices is not None,
    )


def vitb8(dynamic_img_size: bool = True, out_indices: int | List[int] | None = None) -> nn.Module:
    """Initializes the vision transformer ViTB-8 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTB-8 based foundation model.
    """
    return timm.create_model(
        model_name="vit_base_patch8_224",
        dynamic_img_size=dynamic_img_size,
        pretrained_cfg={
            "url": f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vitb8.pth",
            "num_classes": 0
        },
        pretrained=True,
        out_indices=out_indices,
        features_only=out_indices is not None,
    )


def vitl14(dynamic_img_size: bool = True, out_indices: int | List[int] | None = None) -> nn.Module:
    """Initializes the vision transformer ViTL-14 pathology FM by kaiko.ai.

    Args:
        dynamic_img_size: Whether to allow the interpolation embedding
            to be interpolated at `forward()` time when image grid changes
            from original.
        out_indices: Weather and which multi-level patch embeddings to return.

    Returns:
        The torch ViTL-14 based foundation model.
    """
    return timm.create_model(
        model_name="vit_large_patch14_reg4_dinov2",
        pretrained_cfg={
            "url": f"https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/{RELEASE_TAG}/vitl14.pth",
            "num_classes": 0,
        },
        pretrained=True,
        out_indices=out_indices,
        dynamic_img_size=dynamic_img_size,
        features_only=out_indices is not None,
    )
