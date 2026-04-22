import os
import json
import re
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

SYSTEM_PROMPT = """You are a precise answer engine. You will receive a query and possibly some context from assets.

CRITICAL RULES:
1. Give the SHORTEST possible correct answer
2. Do NOT add any explanation, reasoning, or extra words
3. Match the expected output format exactly
4. For math questions: compute the answer and state it simply
5. For factual questions: give just the fact
6. Do NOT say "Sure!", "Here's the answer", "I think", etc.
7. Be direct and concise

EXAMPLES OF GOOD ANSWERS:
- Query: "What is 10 + 15?" → "The sum is 25."
- Query: "What is the capital of France?" → "The capital of France is Paris."
- Query: "Summarize this text" → Give a brief 1-2 sentence summary

If asset URLs are provided with context, use that context to answer the query.
"""


async def fetch_asset_content(url: str) -> str:
    """Fetch content from an asset URL."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text" in content_type or "json" in content_type or "html" in content_type:
                return response.text[:5000]  # Limit to avoid token overflow
            else:
                return f"[Binary content from {url}, type: {content_type}]"
    except Exception as e:
        return f"[Failed to fetch {url}: {str(e)}]"


async def call_gemini(query: str, asset_context: str = "") -> str:
    """Call Gemini API and return the response text."""
    
    user_message = query
    if asset_context:
        user_message = f"Context from provided assets:\n{asset_context}\n\nQuery: {query}"
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{SYSTEM_PROMPT}\n\nQuery: {user_message}"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                GEMINI_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
            
            return "Unable to generate response."
    
    except Exception as e:
        return f"Error: {str(e)}"


@app.post("/api/answer")
async def answer(request: Request):
    """Main endpoint that receives queries and returns answers."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"output": "Invalid JSON in request body."}
        )
    
    query = body.get("query", "")
    assets = body.get("assets", [])
    
    if not query:
        return JSONResponse(
            status_code=400,
            content={"output": "No query provided."}
        )
    
    # Fetch asset content if URLs provided
    asset_context = ""
    if assets:
        contents = []
        for url in assets:
            if url and isinstance(url, str) and url.startswith("http"):
                content = await fetch_asset_content(url)
                contents.append(f"--- Content from {url} ---\n{content}")
        if contents:
            asset_context = "\n\n".join(contents)
    
    # Call Gemini
    answer_text = await call_gemini(query, asset_context)
    
    return {"output": answer_text}


@app.get("/")
async def health():
    """Health check endpoint."""
    return {"status": "alive", "message": "Hackathon API is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
