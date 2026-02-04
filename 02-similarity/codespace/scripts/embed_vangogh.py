"""
Script to generate embeddings for Van Gogh paintings.
"""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from embeddings import get_embeddings_for_image_and_text


def process_image_with_embedding(image_data):
    """
    Process a single image to get its embedding.

    Args:
        image_data: Tuple of (index, image_path)

    Returns:
        Dictionary with image index, filename, and embedding
    """
    idx, image_path = image_data
    # Get embedding without text (text parameter is optional)
    embedding = get_embeddings_for_image_and_text(str(image_path), text="")
    return {
        "index": idx,
        "filename": image_path.name,
        "path": str(image_path.relative_to(Path(__file__).parent.parent)),
        "embedding": embedding
    }


def main():
    # Define the directory path
    data_dir = Path(__file__).parent.parent / 'data' / 'VanGogh'

    # Get all image files
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = set()
    for pattern in image_patterns:
        image_files.update(data_dir.glob(pattern))

    # Sort for consistent ordering
    image_files = sorted(image_files)

    print(f"Found {len(image_files)} images in {data_dir}")
    print("\nImages:")
    print("-" * 80)
    for img in image_files:
        print(f"  - {img.name}")

    # Generate embeddings with multithreading
    print(f"\n\nGenerating embeddings with 5 threads...")
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_image_with_embedding, (i, img)): i
            for i, img in enumerate(image_files)
        }

        # Process completed tasks with progress bar
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Processing images"):
            result = future.result()
            results.append(result)

    # Sort results by index to maintain order
    results.sort(key=lambda x: x["index"])

    # Save to JSON file
    output_file = data_dir / 'VanGogh_embeddings.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(results)} images with embeddings to {output_file}")

    return results


if __name__ == "__main__":
    results = main()
