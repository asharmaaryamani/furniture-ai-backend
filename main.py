# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pinecone
from sentence_transformers import SentenceTransformer
import requests
import os
import uvicorn

# ------------------------------
# 1️⃣ App Initialization
# ------------------------------
app = FastAPI(title="Furniture AI Recommendation API")

# ------------------------------
# 2️⃣ Environment Variables (you'll set these in Render)
# ------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "furniture-ai")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")  # optional (for Hugging Face API)

# ------------------------------
# 3️⃣ Initialize Pinecone + Model
# ------------------------------
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)
embedder = SentenceTransformer(MODEL_NAME)

# ------------------------------
# 4️⃣ Request Schemas
# ------------------------------
class QueryInput(BaseModel):
    query: str
    top_k: Optional[int] = 5

class GenInput(BaseModel):
    title: str
    brand: Optional[str] = None
    description: Optional[str] = None
    style_prompt: Optional[str] = "creative product description"

# ------------------------------
# 5️⃣ Health Check Endpoint
# ------------------------------
@app.get("/")
def root():
    return {"message": "Furniture AI Backend is running successfully!"}

# ------------------------------
# 6️⃣ Recommendation Endpoint
# ------------------------------
@app.post("/recommend")
def recommend(input: QueryInput):
    try:
        # Encode query using SentenceTransformer
        query_emb = embedder.encode([input.query]).tolist()
        # Query Pinecone for top similar products
        res = index.query(vector=query_emb[0], top_k=input.top_k, include_metadata=True)

        recommendations = []
        for match in res['matches']:
            meta = match['metadata']
            meta['score'] = match['score']
            recommendations.append(meta)

        return {"query": input.query, "recommendations": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# 7️⃣ Generative Description Endpoint
# ------------------------------
@app.post("/generate")
def generate_description(input: GenInput):
    """
    Uses Hugging Face inference API or fallback templating to create creative descriptions.
    """
    prompt = f"""
    Write a {input.style_prompt} for the following furniture product:
    Title: {input.title}
    Brand: {input.brand}
    Description: {input.description}
    """

    if GENAI_API_KEY:
        # Use Hugging Face Inference API
        HF_URL = "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {GENAI_API_KEY}"}
        response = requests.post(HF_URL, headers=headers, json={"inputs": prompt})

        if response.status_code == 200:
            data = response.json()
            text = data[0].get("generated_text", "")
            return {"generated_description": text.strip()}
        else:
            return {"error": f"HuggingFace Error: {response.text}"}

    # Fallback simple generated text
    default_text = (
        f"{input.title} by {input.brand} — "
        f"A premium piece featuring elegant design and modern craftsmanship. "
        f"Ideal for homes seeking comfort, quality, and timeless appeal."
    )
    return {"generated_description": default_text}

# ------------------------------
# 8️⃣ Run Locally (for Colab Testing Only)
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
