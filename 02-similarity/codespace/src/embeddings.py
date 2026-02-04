import cohere
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Cohere client
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

def get_embeddings_for_text(text: str) -> list[float]:
    return co.embed(
        model="embed-v4.0",
        inputs=[{"content": text}],
        input_type="search_document",
        embedding_types=["float"],
        output_dimension=1024,
    ).embeddings.float_[0]


def get_embeddings_for_image_and_text(image_path: str, text: str = None) -> list[float]:
    """
    text parameter is optional
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    data_url = f"data:image/jpeg;base64,{encoded_string}"
    
    example_doc = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    
    return co.embed(
        model="embed-v4.0",
        inputs=[{"content": example_doc}],
        input_type="search_document",
        embedding_types=["float"],
        output_dimension=1024,
    ).embeddings.float_[0]

