"""
Script to load The Republic by Plato and split it into paragraph chunks.
"""

import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from embeddings import get_embeddings_for_text


def load_and_chunk_text(file_path):
    """
    Load the text file and split it into paragraph chunks.

    Args:
        file_path: Path to the text file

    Returns:
        List of text chunks (paragraphs)
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by lines
    lines = content.split('\n')

    # Remove line numbers and arrows (format: "  123→text")
    cleaned_lines = []
    for line in lines:
        # Match pattern like "   123→" at the start of line
        match = re.match(r'^\s*\d+→', line)
        if match:
            # Remove the line number and arrow
            cleaned_line = line[match.end():]
            cleaned_lines.append(cleaned_line)
        else:
            cleaned_lines.append(line)

    # Join back into text
    cleaned_text = '\n'.join(cleaned_lines)

    # Split by double newlines (blank lines separate paragraphs)
    paragraphs = re.split(r'\n\s*\n', cleaned_text)

    # Filter out empty paragraphs and strip whitespace
    chunks = [p.strip() for p in paragraphs if p.strip()]

    return chunks


def process_chunk_with_embedding(chunk_data):
    """
    Process a single chunk to get its embedding.

    Args:
        chunk_data: Tuple of (index, chunk_text)

    Returns:
        Dictionary with chunk index, text, and embedding
    """
    idx, chunk = chunk_data
    embedding = get_embeddings_for_text(chunk)
    return {
        "index": idx,
        "text": chunk,
        "embedding": embedding
    }


def main():
    # Define the file path
    data_dir = Path(__file__).parent.parent / 'data' / 'Plato'
    file_path = data_dir / 'TheRepublic.txt'

    # Load and chunk the text
    print("Loading and chunking text...")
    chunks = load_and_chunk_text(file_path)

    # Print statistics
    print(f"Total number of chunks (paragraphs): {len(chunks)}")
    print(f"\nFirst 5 chunks:")
    print("-" * 80)

    for i, chunk in enumerate(chunks[:5], 1):
        print(f"\nChunk {i}:")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print(f"Length: {len(chunk)} characters")

    # Generate embeddings with multithreading
    print(f"\n\nGenerating embeddings with 5 threads...")
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_chunk_with_embedding, (i, chunk)): i
            for i, chunk in enumerate(chunks)
        }

        # Process completed tasks with progress bar
        for future in tqdm(as_completed(futures), total=len(chunks), desc="Processing chunks"):
            result = future.result()
            results.append(result)

    # Sort results by index to maintain order
    results.sort(key=lambda x: x["index"])

    # Save to JSON file
    output_file = data_dir / 'TheRepublic_embeddings.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(results)} chunks with embeddings to {output_file}")

    return results


if __name__ == "__main__":
    chunks = main()
