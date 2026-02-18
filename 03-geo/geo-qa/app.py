#!/usr/bin/env python3
"""
OlmoEarth City Embeddings Web App
A Flask application for processing satellite images through OlmoEarth and visualizing
embeddings against GDP data.
"""

import os
import json
import csv
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

app = Flask(__name__)
app.config["SECRET_KEY"] = "olmoearth-secret-key"

# Data storage
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.json")
CITIES_CSV = os.path.join(os.path.dirname(__file__), "cities_gdp.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load OlmoEarth model (lazy loading)
model = None
device = None


def get_model():
    """Lazy load the OlmoEarth model."""
    global model, device
    if model is None:
        print("Loading OlmoEarth-nano model...")
        model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded on {device}")
    return model, device


def load_cities():
    """Load cities from CSV file."""
    cities = []
    with open(CITIES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Clean numeric values (remove commas)
                gdp_billion = row["gdp_billion_usd"].replace(",", "")
                population = row["population"].replace(",", "")
                gdp_per_capita = row["gdp_per_capita"].replace(",", "")

                cities.append(
                    {
                        "city": row["city"],
                        "country": row["country"],
                        "gdp_billion_usd": float(gdp_billion),
                        "population": int(float(population)),
                        "gdp_per_capita": float(gdp_per_capita),
                    }
                )
            except (ValueError, KeyError) as e:
                print(
                    f"Warning: Skipping row with invalid data: {row.get('city', 'Unknown')} - {e}"
                )
                continue
    return cities


def load_embeddings():
    """Load stored embeddings from JSON file."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_embeddings(embeddings):
    """Save embeddings to JSON file."""
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f, indent=2)


def download_image(url):
    """Download image from URL."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img


def prepare_image_for_olmoearth(img, target_size=64):
    """Convert RGB image to pseudo-Sentinel-2 format."""
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize maintaining aspect ratio, then center crop
    img.thumbnail((target_size, target_size), Image.BILINEAR)
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - img.width) // 2
    offset_y = (target_size - img.height) // 2
    new_img.paste(img, (offset_x, offset_y))
    img = new_img

    # Convert to numpy and normalize
    rgb = np.array(img).astype(np.float32) / 255.0

    # Create 12-band Sentinel-2-like array (BHWTC format)
    sentinel2 = np.zeros((1, target_size, target_size, 1, 12), dtype=np.float32)
    sentinel2[0, :, :, 0, 0] = rgb[:, :, 2]  # Blue → B02
    sentinel2[0, :, :, 0, 1] = rgb[:, :, 1]  # Green → B03
    sentinel2[0, :, :, 0, 2] = rgb[:, :, 0]  # Red → B04

    return sentinel2


def run_olmoearth_inference(image_array):
    """Run OlmoEarth inference on prepared image and extract attention maps."""
    model, device = get_model()

    # Convert to tensor
    sentinel2_tensor = torch.tensor(image_array, dtype=torch.float32, device=device)
    mask = (
        torch.ones((1, 64, 64, 1, 3), dtype=torch.float32, device=device)
        * MaskValue.ONLINE_ENCODER.value
    )
    timestamps = torch.tensor([[[15, 6, 2024]]], device=device)

    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=sentinel2_tensor,
        sentinel2_l2a_mask=mask,
        timestamps=timestamps,
    )

    attention_maps = []

    with torch.no_grad():
        output = model.encoder(sample, fast_pass=True, patch_size=4)
        features = output["tokens_and_masks"].sentinel2_l2a
        pooled = features.mean(dim=[3, 4])  # (1, 16, 16, 128)

        # Try to extract attention from the model if available
        # The encoder outputs may contain attention weights in different layers
        try:
            # Check if attention weights are available in output
            if hasattr(output, 'attentions') or 'attentions' in output:
                attn_data = output.get('attentions') if isinstance(output, dict) else output.attentions
                # Extract attention from multiple heads if available
                if attn_data is not None:
                    for layer_attn in attn_data:
                        # Average over batch and sequence dims, keep head dimension
                        attn_map = layer_attn.mean(dim=(0, 2))  # Average batch and query positions
                        attention_maps.append(attn_map.cpu().numpy().tolist())
        except Exception as e:
            print(f"Note: Could not extract attention maps - {e}")

    # Convert to numpy and flatten spatial dimensions
    embedding = pooled.cpu().numpy()
    # Average over spatial dimensions to get single 128-dim vector
    embedding = embedding.mean(axis=(1, 2))[0]  # (128,)

    return embedding.tolist(), attention_maps


@app.route("/")
def index():
    """Home page with city selection form."""
    cities = load_cities()
    embeddings = load_embeddings()

    # Get list of cities that already have embeddings
    processed_cities = list(embeddings.keys())

    # Filter to get only unprocessed cities
    unprocessed_cities = [c for c in cities if c["city"] not in processed_cities]

    return render_template(
        "index.html",
        cities=cities,
        processed_cities=processed_cities,
        unprocessed_cities=unprocessed_cities,
        total_cities=len(cities),
    )


@app.route("/process", methods=["POST"])
def process():
    """Process image through OlmoEarth."""
    city = request.form.get("city")
    image_url = request.form.get("image_url")

    if not city or not image_url:
        return jsonify({"error": "City and image URL are required"}), 400

    try:
        # Download and process image
        print(f"Processing {city}...")
        img = download_image(image_url)
        image_array = prepare_image_for_olmoearth(img)

        # Run inference
        embedding, attention_maps = run_olmoearth_inference(image_array)

        # Load city data
        cities = load_cities()
        city_data = next((c for c in cities if c["city"] == city), None)

        # Save embedding
        embeddings = load_embeddings()
        embeddings[city] = {
            "embedding": embedding,
            "attention_maps": attention_maps,
            "image_url": image_url,
            "gdp_per_capita": city_data["gdp_per_capita"] if city_data else None,
            "gdp_billion_usd": city_data["gdp_billion_usd"] if city_data else None,
            "population": city_data["population"] if city_data else None,
        }
        save_embeddings(embeddings)

        return jsonify(
            {
                "success": True,
                "city": city,
                "embedding_dim": len(embedding),
                "embedding_sample": embedding[:5],  # First 5 values
                "attention_maps_extracted": len(attention_maps) > 0,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
def dashboard():
    """Dashboard page for visualizing embeddings vs GDP."""
    embeddings = load_embeddings()

    if not embeddings:
        return render_template(
            "dashboard.html",
            has_data=False,
            message="No embeddings processed yet. Go to the home page to add cities.",
        )

    # Prepare data for visualization
    cities_with_data = []
    for city, data in embeddings.items():
        if data.get("gdp_per_capita") and data.get("embedding"):
            cities_with_data.append(
                {
                    "city": city,
                    "country": data.get("country", ""),
                    "gdp_per_capita": data["gdp_per_capita"],
                    "gdp_billion_usd": data.get("gdp_billion_usd", 0),
                    "population": data.get("population", 0),
                    "embedding": data["embedding"],
                }
            )

    return render_template(
        "dashboard.html", has_data=True, cities=cities_with_data, embedding_dims=128
    )


@app.route("/api/cities")
def api_cities():
    """API endpoint to get list of cities."""
    cities = load_cities()
    return jsonify(cities)


@app.route("/api/embeddings")
def api_embeddings():
    """API endpoint to get all embeddings."""
    embeddings = load_embeddings()
    return jsonify(embeddings)


@app.route("/vote")
def vote():
    """Vote page for comparing embedding dimensions."""
    embeddings = load_embeddings()

    if len(embeddings) < 2:
        return render_template(
            "vote.html",
            has_data=False,
            message="Need at least 2 cities with embeddings to vote. Add more cities first.",
        )

    import random

    # Get two random dimensions to compare
    dim1, dim2 = random.sample(range(128), 2)

    # Prepare data for both dimensions
    cities_data = []
    for city, data in embeddings.items():
        if data.get("gdp_per_capita") and data.get("embedding"):
            cities_data.append(
                {
                    "city": city,
                    "country": data.get("country", ""),
                    "gdp_per_capita": data["gdp_per_capita"],
                    "embedding": data["embedding"],
                }
            )

    return render_template("vote.html", has_data=True, dim1=dim1, dim2=dim2, cities=cities_data)


@app.route("/vote/cast", methods=["POST"])
def cast_vote():
    """Record a vote for which dimension is better."""
    data = request.get_json()
    winner = data.get("winner")
    loser = data.get("loser")

    if winner is None or loser is None:
        return jsonify({"error": "Winner and loser dimensions required"}), 400

    # Load existing votes
    votes_file = os.path.join(DATA_DIR, "votes.json")
    votes = {}
    if os.path.exists(votes_file):
        with open(votes_file, "r") as f:
            votes = json.load(f)

    # Record vote
    if str(winner) not in votes:
        votes[str(winner)] = {"wins": 0, "losses": 0, "total": 0}
    if str(loser) not in votes:
        votes[str(loser)] = {"wins": 0, "losses": 0, "total": 0}

    votes[str(winner)]["wins"] += 1
    votes[str(winner)]["total"] += 1
    votes[str(loser)]["losses"] += 1
    votes[str(loser)]["total"] += 1

    # Save votes
    with open(votes_file, "w") as f:
        json.dump(votes, f, indent=2)

    return jsonify({"success": True})


@app.route("/leaderboard")
def leaderboard():
    """Leaderboard page showing best embedding dimensions."""
    votes_file = os.path.join(DATA_DIR, "votes.json")
    votes = {}
    if os.path.exists(votes_file):
        with open(votes_file, "r") as f:
            votes = json.load(f)

    # Calculate scores for each dimension
    leaderboard_data = []
    for dim, data in votes.items():
        total = data.get("total", 0)
        wins = data.get("wins", 0)
        if total > 0:
            win_rate = wins / total
            leaderboard_data.append(
                {
                    "dimension": int(dim),
                    "wins": wins,
                    "losses": data.get("losses", 0),
                    "total": total,
                    "win_rate": win_rate,
                }
            )

    # Sort by win rate (descending), then by total votes
    leaderboard_data.sort(key=lambda x: (-x["win_rate"], -x["total"]))

    return render_template("leaderboard.html", leaderboard=leaderboard_data)


# Attention voting routes
@app.route("/attention-vote")
def attention_vote():
    """Vote page for comparing attention heads."""
    embeddings = load_embeddings()

    if len(embeddings) < 1:
        return render_template(
            "attention_vote.html",
            has_data=False,
            message="Need at least 1 city with embeddings to vote on attention. Add cities first.",
        )

    import random

    # Get a random city
    city_name = random.choice(list(embeddings.keys()))
    city_data = embeddings[city_name]

    # Check if we have attention maps
    attention_maps = city_data.get("attention_maps", [])

    if attention_maps:
        # Use real attention maps - select 2 random layers/heads
        max_head_idx = len(attention_maps) - 1
        if max_head_idx < 1:
            head1, head2 = 0, 0
        else:
            head1, head2 = random.sample(range(len(attention_maps)), min(2, len(attention_maps)))
        has_real_attention = True
    else:
        # Fall back to synthetic patterns if no real attention
        head1, head2 = random.sample(range(8), 2)
        has_real_attention = False

    return render_template(
        "attention_vote.html",
        has_data=True,
        city=city_name,
        image_url=city_data.get("image_url", ""),
        head1=head1,
        head2=head2,
        has_real_attention=has_real_attention,
        attention_maps=attention_maps,
    )


@app.route("/attention-vote/cast", methods=["POST"])
def cast_attention_vote():
    """Record a vote for which attention head is better."""
    data = request.get_json()
    winner = data.get("winner")
    loser = data.get("loser")
    city = data.get("city")

    if winner is None or loser is None:
        return jsonify({"error": "Winner and loser heads required"}), 400

    # Load existing votes
    attention_votes_file = os.path.join(DATA_DIR, "attention_votes.json")
    votes = {}
    if os.path.exists(attention_votes_file):
        with open(attention_votes_file, "r") as f:
            votes = json.load(f)

    # Record vote
    if str(winner) not in votes:
        votes[str(winner)] = {"wins": 0, "losses": 0, "total": 0, "cities": []}
    if str(loser) not in votes:
        votes[str(loser)] = {"wins": 0, "losses": 0, "total": 0, "cities": []}

    votes[str(winner)]["wins"] += 1
    votes[str(winner)]["total"] += 1
    votes[str(winner)]["cities"].append(city)
    votes[str(loser)]["losses"] += 1
    votes[str(loser)]["total"] += 1
    votes[str(loser)]["cities"].append(city)

    # Save votes
    with open(attention_votes_file, "w") as f:
        json.dump(votes, f, indent=2)

    return jsonify({"success": True})


@app.route("/attention-leaderboard")
def attention_leaderboard():
    """Leaderboard page showing best attention heads."""
    attention_votes_file = os.path.join(DATA_DIR, "attention_votes.json")
    votes = {}
    if os.path.exists(attention_votes_file):
        with open(attention_votes_file, "r") as f:
            votes = json.load(f)

    # Calculate scores for each head
    leaderboard_data = []
    for head, data in votes.items():
        total = data.get("total", 0)
        wins = data.get("wins", 0)
        if total > 0:
            win_rate = wins / total
            cities_voted = list(set(data.get("cities", [])))
            leaderboard_data.append(
                {
                    "head": int(head),
                    "wins": wins,
                    "losses": data.get("losses", 0),
                    "total": total,
                    "win_rate": win_rate,
                    "cities_count": len(cities_voted),
                }
            )

    # Sort by win rate (descending), then by total votes
    leaderboard_data.sort(key=lambda x: (-x["win_rate"], -x["total"]))

    return render_template("attention_leaderboard.html", leaderboard=leaderboard_data)


def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("OlmoEarth City Embeddings Web App")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Cities CSV: {CITIES_CSV}")

    # Load cities to verify
    cities = load_cities()
    print(f"Loaded {len(cities)} cities from CSV")

    # Check for existing embeddings
    embeddings = load_embeddings()
    print(f"Existing embeddings: {len(embeddings)} cities")

    print("\n" + "=" * 60)
    print("Starting server...")
    print("URL: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Run Flask app
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
