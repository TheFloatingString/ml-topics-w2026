#!/usr/bin/env python3
"""
Try running OlmoEarth-nano on a Wikipedia satellite image.
"""

import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue


def download_image(url):
    """Download image from URL."""
    print(f"Downloading image from: {url}")

    # Try with headers to avoid 403
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        print(f"Downloaded image: {img.size}, mode={img.mode}")
        return img
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print(f"403 Forbidden - creating synthetic test image instead")
            return create_synthetic_image()
        raise


def create_synthetic_image(size=256):
    """Create a synthetic satellite-like image for testing."""
    print(f"Creating synthetic {size}x{size} test image...")

    # Create a gradient pattern that looks somewhat like land/water
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)

    # Create RGB channels with some variation
    r = (0.3 + 0.4 * xx + 0.2 * np.sin(yy * 10)) * 255
    g = (0.4 + 0.3 * yy + 0.2 * np.cos(xx * 8)) * 255
    b = (0.2 + 0.5 * (1 - xx) + 0.1 * np.sin((xx + yy) * 5)) * 255

    # Stack and convert to uint8
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

    img = Image.fromarray(rgb)
    print(f"Created synthetic image: {img.size}, mode={img.mode}")
    return img


def prepare_for_olmoearth(img, target_size=32):
    """
    Convert RGB image to pseudo-Sentinel-2 format.
    Maps RGB to B04(Red), B03(Green), B02(Blue) and pads rest with zeros.
    Uses 32x32 to match nano model's expected input size.
    """
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to target size (center crop to maintain aspect ratio)
    img.thumbnail((target_size, target_size), Image.BILINEAR)

    # Create a square image by centering the resized image
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - img.width) // 2
    offset_y = (target_size - img.height) // 2
    new_img.paste(img, (offset_x, offset_y))
    img = new_img

    # Convert to numpy array (H, W, 3)
    rgb = np.array(img).astype(np.float32)

    # Normalize to [0, 1] range
    rgb = rgb / 255.0

    # Create 12-band Sentinel-2-like array
    # Sentinel-2 band order: B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
    sentinel2 = np.zeros((target_size, target_size, 12), dtype=np.float32)

    # Map RGB to corresponding Sentinel-2 bands
    # B02 = Blue (index 0), B03 = Green (index 1), B04 = Red (index 2)
    sentinel2[:, :, 0] = rgb[:, :, 2]  # Blue → B02
    sentinel2[:, :, 1] = rgb[:, :, 1]  # Green → B03
    sentinel2[:, :, 2] = rgb[:, :, 0]  # Red → B04

    # Note: Bands 3-11 remain zeros (no data)

    # Convert to BHWTC format: (Batch, Height, Width, Time, Channels)
    # Input needs to be (B, H, W, T, C) = (1, 32, 32, 1, 12)
    sentinel2 = sentinel2.transpose(2, 0, 1)  # (12, 32, 32)
    sentinel2 = sentinel2[None, :, :, None, :]  # (1, 12, 32, 1, 32) - wrong!

    # Fix: need (1, 32, 32, 1, 12)
    sentinel2 = np.zeros((1, target_size, target_size, 1, 12), dtype=np.float32)
    sentinel2[0, :, :, 0, 0] = rgb[:, :, 2]  # Blue → B02
    sentinel2[0, :, :, 0, 1] = rgb[:, :, 1]  # Green → B03
    sentinel2[0, :, :, 0, 2] = rgb[:, :, 0]  # Red → B04

    return sentinel2


def run_inference():
    """Run OlmoEarth-nano on Wikipedia satellite image."""
    # Wikipedia satellite image of Montreal
    url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Montr%C3%A9al_Satellite.jpg"

    try:
        # Download and prepare image
        img = download_image(url)
        sentinel2_data = prepare_for_olmoearth(img, target_size=64)

        print(f"\nPrepared input shape: {sentinel2_data.shape}")
        print(
            f"Non-zero bands: {np.sum(np.any(sentinel2_data[0, :, :, 0, :] != 0, axis=(0, 1)))}/12"
        )

        # Load model
        print("\nLoading OlmoEarth-nano model...")
        model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded on {device}")

        # Prepare tensors
        sentinel2_tensor = torch.tensor(
            sentinel2_data, dtype=torch.float32, device=device
        )
        mask = (
            torch.ones((1, 64, 64, 1, 3), dtype=torch.float32, device=device)
            * MaskValue.ONLINE_ENCODER.value
        )
        timestamps = torch.tensor([[[15, 6, 2024]]], device=device)  # day, month, year

        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=sentinel2_tensor,
            sentinel2_l2a_mask=mask,
            timestamps=timestamps,
        )

        # Run inference
        print("\nRunning inference...")
        with torch.no_grad():
            output = model.encoder(sample, fast_pass=True, patch_size=4)
            features = output["tokens_and_masks"].sentinel2_l2a

        print(f"\nOutput shape: {features.shape}")
        print(f"Feature dimensions: {features.shape[-1]}")

        # Pool features
        pooled = features.mean(dim=[3, 4])
        print(f"Pooled features shape: {pooled.shape}")

        # Check feature statistics
        print(f"\nFeature statistics:")
        print(f"  Mean: {pooled.mean().item():.4f}")
        print(f"  Std: {pooled.std().item():.4f}")
        print(f"  Min: {pooled.min().item():.4f}")
        print(f"  Max: {pooled.max().item():.4f}")

        # Print actual embedding values
        print(f"\n{'=' * 60}")
        print("EMBEDDING VALUES (first 5 patches, first 10 dimensions):")
        print(f"{'=' * 60}")

        # Get first 5 patches (out of 16x16=256)
        for i in range(min(5, pooled.shape[1])):
            for j in range(min(5, pooled.shape[2])):
                patch_embedding = pooled[0, i, j, :].cpu().numpy()
                print(f"\nPatch [{i},{j}] - First 10 dims: {patch_embedding[:10]}")
                print(f"           L2 norm: {np.linalg.norm(patch_embedding):.4f}")

        print(f"\n{'=' * 60}")
        print("SAMPLE COMPARISON (cosine similarity between patches):")
        print(f"{'=' * 60}")

        # Compare a few patches
        from numpy import dot
        from numpy.linalg import norm

        def cosine_similarity(a, b):
            return dot(a, b) / (norm(a) * norm(b))

        patch_0_0 = pooled[0, 0, 0, :].cpu().numpy()
        patch_0_1 = pooled[0, 0, 1, :].cpu().numpy()
        patch_1_0 = pooled[0, 1, 0, :].cpu().numpy()
        patch_8_8 = pooled[0, 8, 8, :].cpu().numpy()

        print(
            f"\nCosine similarity (0,0) vs (0,1): {cosine_similarity(patch_0_0, patch_0_1):.4f}"
        )
        print(
            f"Cosine similarity (0,0) vs (1,0): {cosine_similarity(patch_0_0, patch_1_0):.4f}"
        )
        print(
            f"Cosine similarity (0,0) vs (8,8): {cosine_similarity(patch_0_0, patch_8_8):.4f}"
        )

        print(f"\n{'=' * 60}")
        print("[OK] Inference completed successfully!")
        print(f"{'=' * 60}")
        print("\nNote: Embeddings are lower quality because only 3/12 bands have data")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_inference()
