import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """You are a precise answer engine.

RULES:
1. Give the SHORTEST possible correct answer
2. Do NOT add explanation, reasoning, or extra words
3. For math: compute and state simply (e.g. "The sum is 25.")
4. For facts: state the fact directly
5. Never say "Sure!", "Here's the answer", "I think", etc.
6. Be direct and concise

If context from assets is provided, use it to answer the query.
"""


async def fetch_asset_content(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if any(t in ct for t in ["text", "json", "html"]):
                return resp.text[:5000]
            return f"[Binary content from {url}]"
    except Exception as e:
        return f"[Failed to fetch {url}: {e}]"


async def call_llm(query: str, asset_context: str = "") -> str:
    user_msg = query
    if asset_context:
        user_msg = f"Context from assets:\n{asset_context}\n\nQuery: {query}"

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT}\n\nQuery: {user_msg}"}]}
        ],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024}
    }

    # Try primary model, fallback to lite — no retries, just move on
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]

    async with httpx.AsyncClient(timeout=18.0) as client:
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 429:
                    continue  # try next model immediately, no waiting
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            except Exception:
                continue

    return "Unable to process query."


@app.post("/api/answer")
async def answer(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"output": "Invalid request."})

    query = body.get("query", "")
    assets = body.get("assets", [])

    if not query:
        return JSONResponse(status_code=400, content={"output": "No query provided."})

    asset_context = ""
    if assets:
        contents = []
        for url in assets:
            if url and isinstance(url, str) and url.startswith("http"):
                content = await fetch_asset_content(url)
                contents.append(content)
        if contents:
            asset_context = "\n\n".join(contents)

    result = await call_llm(query, asset_context)
    return {"output": result}


@app.get("/")
async def health():
    return {"status": "ok"}
