"""
FastAPI application for comparing image and text embeddings.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


app = FastAPI(title="Art & Philosophy Similarity Explorer")

# Load embedding data
DATA_DIR = Path(__file__).parent / "data"

# Load Plato embeddings
with open(DATA_DIR / "Plato" / "TheRepublic_embeddings.json", "r", encoding="utf-8") as f:
    plato_data = json.load(f)

# Load Van Gogh embeddings
with open(DATA_DIR / "VanGogh" / "VanGogh_embeddings.json", "r", encoding="utf-8") as f:
    vangogh_data = json.load(f)

# Load Picasso embeddings
with open(DATA_DIR / "Picasso" / "Picasso_embeddings.json", "r", encoding="utf-8") as f:
    picasso_data = json.load(f)

# Combine image data
image_data = []
for item in vangogh_data:
    image_data.append({
        "id": f"vangogh_{item['index']}",
        "artist": "Van Gogh",
        "filename": item['filename'],
        "path": item['path'].replace("\\", "/"),
        "embedding": item['embedding']
    })

for item in picasso_data:
    image_data.append({
        "id": f"picasso_{item['index']}",
        "artist": "Picasso",
        "filename": item['filename'],
        "path": item['path'].replace("\\", "/"),
        "embedding": item['embedding']
    })


class SimilarityRequest(BaseModel):
    image_id: str
    passage_index: int


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_file = Path(__file__).parent / "templates" / "index.html"
    with open(html_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/images")
async def get_images():
    """Get all image metadata (without embeddings)."""
    return [{
        "id": img["id"],
        "artist": img["artist"],
        "filename": img["filename"],
        "path": img["path"]
    } for img in image_data]


@app.get("/api/passages/random")
async def get_random_passages(count: int = 10):
    """Get random Plato passages."""
    if count > len(plato_data):
        count = len(plato_data)

    selected = random.sample(plato_data, count)
    return [{
        "index": item["index"],
        "text": item["text"]
    } for item in selected]


@app.post("/api/similarity")
async def calculate_similarity(request: SimilarityRequest):
    """Calculate cosine similarity between an image and a passage."""
    # Find the image
    image = next((img for img in image_data if img["id"] == request.image_id), None)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Find the passage
    passage = next((p for p in plato_data if p["index"] == request.passage_index), None)
    if not passage:
        raise HTTPException(status_code=404, detail="Passage not found")

    # Calculate similarity
    similarity = cosine_similarity(image["embedding"], passage["embedding"])

    return {
        "image_id": request.image_id,
        "image_filename": image["filename"],
        "artist": image["artist"],
        "passage_index": request.passage_index,
        "passage_text": passage["text"],
        "similarity": similarity
    }


# Mount static files
app.mount("/data", StaticFiles(directory="data"), name="data")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
