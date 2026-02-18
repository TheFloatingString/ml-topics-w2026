# OlmoEarth City Embeddings Web App

A Flask web application that processes satellite images through AI2's OlmoEarth model and visualizes the relationship between city economic indicators (GDP) and image embeddings.

## Features

- **City Selection**: Choose from 977 cities with GDP data
- **Image Processing**: Submit satellite image URLs for OlmoEarth-nano processing
- **Embedding Storage**: Automatically stores 128-dimensional embeddings for each city
- **Interactive Dashboard**: Visualize GDP vs embedding dimensions with Plotly
- **Real-time Updates**: Dynamic chart updates when changing axis selections

## Installation

### Prerequisites

- Python 3.12+
- uv (recommended) or pip

### Setup

1. Clone or navigate to the project directory:
```bash
cd olmoearth-webapp
```

2. Install dependencies with uv:
```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

3. Ensure the cities CSV file is present:
```bash
# Should already exist from previous scraping
cities_gdp.csv
```

## Running the Application

### Using uv (recommended):
```bash
uv run python app.py
```

### Using Python directly:
```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

### 1. Add a City

1. Navigate to the home page (`/`)
2. Select a city from the dropdown (977 cities available)
3. Enter a satellite image URL (e.g., from Google Earth, Sentinel Hub, etc.)
4. Click "Process Image"
5. Wait for OlmoEarth-nano to generate the 128-dimensional embedding

**Example Image URLs:**
- Google Earth Engine thumbnails
- Sentinel Hub WMS services
- Direct satellite imagery URLs

### 2. View Dashboard

1. Navigate to the dashboard (`/dashboard`)
2. Select X-axis: GDP Per Capita, Total GDP, or Population
3. Select Y-axis: Any of the 128 embedding dimensions
4. Choose scale: Linear or Logarithmic
5. Hover over points to see city details

### 3. API Endpoints

- `GET /api/cities` - List all cities with GDP data
- `GET /api/embeddings` - Get all stored embeddings
- `POST /process` - Process a new image (form data: city, image_url)

## Data Format

### Cities CSV (`cities_gdp.csv`)
```csv
city,country,gdp_billion_usd,population,gdp_per_capita
New York,United States,2104,19000000,110737
Tokyo,Japan,1520,37000000,41081
...
```

### Embeddings JSON (`data/embeddings.json`)
```json
{
  "New York": {
    "embedding": [0.123, -0.456, ...],  // 128 values
    "image_url": "https://...",
    "gdp_per_capita": 110737,
    "gdp_billion_usd": 2104,
    "population": 19000000
  }
}
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Browser  │────▶│   Flask Server   │────▶│  OlmoEarth-nano │
│                 │     │                  │     │    (AI2 Model)  │
│  - Select city  │     │  - Load cities   │     │                 │
│  - Submit URL   │     │  - Process img │     │  - 128-dim emb  │
│  - View charts  │◀────│  - Store data  │◀────│                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │
         │              ┌────────┴────────┐
         │              │   Data Files    │
         └─────────────▶│  - cities.csv   │
                        │  - embeddings   │
                        │     .json       │
                        └─────────────────┘
```

## Technical Details

### OlmoEarth Processing
- **Model**: OlmoEarth-v1-Nano (3.5M parameters)
- **Input**: 64×64 RGB image → mapped to 12 Sentinel-2 bands
- **Output**: 128-dimensional embedding vector
- **Processing**: ~1-2 seconds per image on CPU

### Visualization
- **Library**: Plotly.js
- **Chart Types**: Scatter plot with color-coded points
- **Interactivity**: Hover tooltips, zoom, pan
- **Scales**: Linear or Logarithmic

## Troubleshooting

### Model Loading Issues
```bash
# If olmoearth_pretrain fails to load
uv pip install --force-reinstall olmoearth-pretrain
```

### Image Download Failures
- Ensure the URL is publicly accessible
- Check that the URL points directly to an image (not a webpage)
- Some sites block automated requests (403 errors)

### Memory Issues
- The nano model uses ~100MB RAM
- Each embedding is 128 floats (~0.5KB)
- For 1000 cities: ~50MB storage

## Future Enhancements

- [ ] Support for multiple images per city
- [ ] Time-series analysis (multiple years)
- [ ] PCA/t-SNE visualization of embeddings
- [ ] Clustering analysis
- [ ] Export to CSV/Excel
- [ ] User authentication
- [ ] Cloud storage integration

## License

This project uses:
- OlmoEarth model: [OlmoEarth Artifact License](https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE)
- Flask: BSD License
- Plotly: MIT License

## Contact

For issues or questions about this web app, please refer to the OlmoEarth documentation at https://github.com/allenai/olmoearth_pretrain