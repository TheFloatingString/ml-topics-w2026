#!/usr/bin/env python3
"""
Inference script for OlmoEarth-nano model.
"""

import torch
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue


def run_inference():
    """Run inference on OlmoEarth-nano with synthetic data."""
    print("Loading OlmoEarth-nano model...")

    # Load model from HuggingFace
    model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Model loaded on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create synthetic input (B=1, H=64, W=64, T=1, C=12 for Sentinel-2)
    print("\nCreating synthetic input...")
    dummy_image = torch.randn(1, 64, 64, 1, 12, device=device)
    dummy_mask = (
        torch.ones(1, 64, 64, 1, 3, device=device) * MaskValue.ONLINE_ENCODER.value
    )
    dummy_timestamps = torch.tensor(
        [[[15, 6, 2024]]], device=device
    )  # day, month (0-indexed), year

    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=dummy_image,
        sentinel2_l2a_mask=dummy_mask,
        timestamps=dummy_timestamps,
    )

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model.encoder(sample, fast_pass=True, patch_size=4)
        features = output["tokens_and_masks"].sentinel2_l2a

    print(f"\nOutput shape: {features.shape}")  # (B, H', W', T, S, D)
    print(f"Feature dimensions: {features.shape[-1]}")

    # Pool features over timestep and band set dimensions
    pooled = features.mean(dim=[3, 4])
    print(f"Pooled features shape: {pooled.shape}")  # (B, H', W', D)

    print("\nInference completed successfully!")


if __name__ == "__main__":
    run_inference()
