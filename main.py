import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

SYSTEM_PROMPT = """You are a precise answer engine that matches expected output formats exactly.

RULES:
1. For math questions: state the operation and result naturally. Example: "What is 10 + 15?" → "The sum is 25."
2. For factual questions: give a complete sentence. Example: "What is the capital of France?" → "The capital of France is Paris."
3. Always respond in complete sentences with proper punctuation.
4. Do NOT add extra explanation, reasoning, caveats, or multiple sentences unless needed.
5. Never say "Sure!", "Here's the answer", "I think", "Let me", etc.
6. Do NOT use markdown, bullet points, or formatting.
7. One concise sentence answer only.

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


async def call_llm(query: str, asset_context: str = "") -> dict:
    user_msg = query
    if asset_context:
        user_msg = f"Context from assets:\n{asset_context}\n\nQuery: {query}"

    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_msg}
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    debug_info = []

    async with httpx.AsyncClient(timeout=18.0) as client:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers
            )
            status = resp.status_code
            debug_info.append(f"model=claude-sonnet-4-5: status={status}")

            if status == 200:
                data = resp.json()
                content = data.get("content", [])
                if content:
                    return {"answer": content[0].get("text", "").strip(), "debug": debug_info}

            debug_info.append(f"body={resp.text[:300]}")

        except Exception as e:
            debug_info.append(f"exception={str(e)}")

    return {"answer": None, "debug": debug_info}


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

    if result["answer"]:
        return {"output": result["answer"]}
    else:
        return {"output": "Unable to process query."}


@app.get("/debug")
async def debug():
    result = await call_llm("What is 2 + 2?")
    return {
        "api_key_set": bool(CLAUDE_API_KEY),
        "api_key_first_10": CLAUDE_API_KEY[:10] + "..." if CLAUDE_API_KEY else "EMPTY",
        "debug": result["debug"],
        "answer": result["answer"]
    }


@app.get("/")
async def health():
    return {"status": "ok"}
