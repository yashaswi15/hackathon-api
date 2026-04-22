import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

SYSTEM_PROMPT = """You are an answer extraction engine. Output ONLY the bare answer with zero extra text.

RULES — read carefully:
- Output the answer ONLY. Nothing before it, nothing after it.
- No punctuation at the end unless it is part of the answer itself.
- No "The answer is", "Result:", "Sure!", or any wrapper.
- No markdown, bullets, asterisks, or formatting.
- Names → bare name only: Bob
- Numbers → bare number only: 25
- Averages/sums/differences → integer if whole, one decimal if not: 85 or 85.5
- Lists → comma-space separated, no trailing comma: Alice, Bob, Charlie
- Yes/No → Yes or No
- True/False → True or False
- Highest/lowest/max/min of names → just the name: Bob
- Highest/lowest/max/min of scores → just the number: 90

If context is provided, use it. Think carefully, then output ONLY the final answer."""


def post_process(text: str) -> str:
    """Strip common Claude artifacts from output."""
    text = text.strip()
    # Remove trailing period if it looks like Claude added one
    if text.endswith(".") and "\n" not in text and len(text.split()) <= 6:
        text = text[:-1]
    # Remove common preamble patterns
    for prefix in [
        "The answer is ", "The result is ", "Answer: ", "Result: ",
        "Output: ", "Response: ", "Sure! ", "Sure, "
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


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
        user_msg = f"Context:\n{asset_context}\n\nQuery: {query}"

    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 256,
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
            debug_info.append(f"status={status}")

            if status == 200:
                data = resp.json()
                content = data.get("content", [])
                if content:
                    raw = content[0].get("text", "").strip()
                    cleaned = post_process(raw)
                    return {"answer": cleaned, "raw": raw, "debug": debug_info}

            debug_info.append(f"body={resp.text[:300]}")

        except Exception as e:
            debug_info.append(f"exception={str(e)}")

    return {"answer": None, "raw": None, "debug": debug_info}


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
    result = await call_llm("Alice scored 80, Bob scored 90. Who scored highest?")
    return {
        "api_key_set": bool(CLAUDE_API_KEY),
        "raw": result.get("raw"),
        "answer": result.get("answer"),
        "debug": result.get("debug"),
    }


@app.get("/")
async def health():
    return {"status": "ok"}
