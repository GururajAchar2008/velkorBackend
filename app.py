"""
app.py — Velkor AI Complete Backend (Single-File Architecture)
Includes: Chat (with streaming, NVIDIA primary -> OpenRouter fallback),
Research search (SearXNG JSON API integration with parallel page reading),
universal file upload/parsing, image generation, and a dedicated
portfolio-assistant route for gururajachar2008.github.io.
"""

import os
import time
import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Generator, List
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests
from datetime import datetime
import traceback
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("velkor-app")

app = Flask(__name__)

# Explicitly configure CORS to support your frontend origin and preflight options
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "https://gururajachar2008.github.io",
                "http://localhost:5173",
                "http://localhost:3000"
            ]
        }
    },
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=False
)
# --- CONFIGURATION ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SEARXNG_URL = os.getenv("SEARXNG_URL", "https://searxng.yourdomain.com").rstrip("/")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hard cap on how much portfolio "context" text the frontend can push into
# the system prompt. Keeps token usage predictable and blocks anyone from
# using this open CORS route to smuggle in an oversized payload.
MAX_CONTEXT_CHARS = 4000
MAX_PORTFOLIO_MESSAGES = 20

if not NVIDIA_API_KEY:
    logger.warning("NVIDIA_API_KEY is not set — NVIDIA NIM calls will be skipped.")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY is not set — the OpenRouter fallback will fail.")
if not os.getenv("SEARXNG_URL"):
    logger.warning("SEARXNG_URL is not set — using default placeholder URL.")

# --- 1. SEARCH + PARALLEL WEB PAGE READING SERVICE (SearXNG + Trafilatura) ---

class SearchService:

    @staticmethod
    def search(query: str, num: int = 8) -> Dict[str, Any]:
        url = f"{SEARXNG_URL}/search"

        try:
            response = requests.get(
                url,
                params={
                    "q": query,
                    "format": "json"
                },
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/151.0.0.0 Safari/537.36"
                    )
                },
                timeout=10,
            )

            if response.status_code != 200:
                logger.warning("SearXNG returned HTTP %s", response.status_code)
                return {"success": False, "results": []}

            data = response.json()
            raw_results = data.get("results", [])

            results = []
            for item in raw_results[:num]:
                results.append({
                    "title": item.get("title", "Untitled"),
                    "snippet": item.get("content", "").strip(),
                    "url": item.get("url", "")
                })

            return {"success": True, "results": results}

        except Exception as e:
            logger.exception("Search failed")
            return {"success": False, "results": [], "error": str(e)}

    @staticmethod
    def fetch_page(url: str, max_chars: int = 10000) -> Dict[str, Any]:
        try:
            import trafilatura

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/151.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }

            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)

            if response.status_code != 200:
                return {"success": False, "url": url, "text": ""}

            raw_html = response.text[:2_000_000]
            extracted = trafilatura.extract(
                raw_html,
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_precision=True
            )

            if not extracted:
                return {"success": False, "url": url, "text": ""}

            extracted = extracted.strip()
            if len(extracted) > max_chars:
                extracted = extracted[:max_chars]
                last_space = extracted.rfind(" ")
                if last_space > max_chars - 500:
                    extracted = extracted[:last_space]
                extracted += "\n[Page content truncated]"

            return {"success": True, "url": url, "text": extracted}

        except Exception as e:
            return {"success": False, "url": url, "text": ""}

    @staticmethod
    def research(query: str, search_count: int = 8, pages_to_read: int = 5) -> Dict[str, Any]:
        search_data = SearchService.search(query, num=search_count)

        if not search_data.get("success"):
            return {"success": False, "sources": [], "context": ""}

        search_results = search_data.get("results", [])
        target_results = search_results[:pages_to_read]

        sources_map = {}
        context_parts = [None] * len(target_results)

        def process_result(index: int, result: Dict[str, Any]):
            title = result.get("title", "Untitled")
            snippet = result.get("snippet", "")
            url = result.get("url", "")

            page = SearchService.fetch_page(url, max_chars=10000)

            if page.get("success") and page.get("text"):
                page_text = page["text"]
                part = f"""
================ SOURCE {index} ================
TITLE:
{title}
URL:
{url}
CONTENT:
{page_text}
================ END SOURCE {index} ================
"""
                source_meta = {"id": index, "title": title, "url": url, "type": "page"}
            else:
                part = f"""
================ SOURCE {index} ================
TITLE:
{title}
URL:
{url}
SEARCH SNIPPET:
{snippet}
================ END SOURCE {index} ================
"""
                source_meta = {"id": index, "title": title, "url": url, "type": "search_snippet"}
            return index - 1, part, source_meta

        with ThreadPoolExecutor(max_workers=pages_to_read) as executor:
            futures = [
                executor.submit(process_result, idx, res)
                for idx, res in enumerate(target_results, start=1)
            ]
            for future in as_completed(futures):
                try:
                    idx_pos, part, source_meta = future.result()
                    context_parts[idx_pos] = part
                    sources_map[idx_pos] = source_meta
                except Exception as e:
                    logger.error("Error processing page in parallel pool: %s", e)

        valid_context_parts = [p for p in context_parts if p is not None]
        ordered_sources = [sources_map[i] for i in sorted(sources_map.keys())]

        return {
            "success": True,
            "sources": ordered_sources,
            "context": "\n".join(valid_context_parts)
        }

# --- 2. DOCUMENT / FILE PARSING SERVICE ---
class DocumentService:
    @staticmethod
    def parse_file(file_storage) -> str:
        filename = file_storage.filename or "upload.bin"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file_storage.save(path)

        extracted_text = ""
        ext = filename.split(".")[-1].lower()
        try:
            if ext in ["txt", "py", "js", "ts", "jsx", "tsx", "html", "css", "json", "md", "csv", "xml", "yaml", "yml", "sql", "sh"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
            elif ext == "pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(path)
                    extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                except Exception:
                    extracted_text = "[PDF parsing library unavailable, binary stored]"
            elif ext == "docx":
                try:
                    from docx import Document
                    doc = Document(path)
                    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                    extracted_text = "\n".join(paragraphs)
                except Exception:
                    extracted_text = "[Error parsing Word document]"
            else:
                extracted_text = f"[Uploaded file: {filename}]"
        except Exception as e:
            extracted_text = f"[Error reading file {filename}]"
        finally:
            if os.path.exists(path):
                os.remove(path)
        return extracted_text

# --- 3. AI PROVIDER ROUTER ---
class AIProviderRouter:
    @staticmethod
    def generate_stream(messages: list) -> Generator[str, None, None]:
        nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers_nvidia = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        }
        payload_nvidia = {
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "messages": messages,
            "temperature": 0.5,
            "stream": True,
        }

        success = False
        if NVIDIA_API_KEY:
            try:
                response = requests.post(nvidia_url, headers=headers_nvidia, json=payload_nvidia, stream=True, timeout=60)
                if response.status_code == 200:
                    success = True
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode("utf-8")
                            if decoded.startswith("data: "):
                                data_str = decoded[6:]
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    chunk_json = json.loads(data_str)
                                    delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if delta:
                                        yield delta
                                except Exception:
                                    pass
                    return
            except Exception as e:
                logger.warning("NVIDIA NIM failed: %s", e)

        # Fallback: OpenRouter Free Tier
        if not success:
            if not OPENROUTER_API_KEY:
                yield "I can't reach the AI providers. Please configure your API keys."
                return

            or_url = "https://openrouter.ai/api/v1/chat/completions"
            headers_or = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://gururajachar2008.github.io/Velkor",
                "X-Title": "Velkor AI",
            }
            payload_or = {
                "model": "deepseek/deepseek-r1-0528:free",
                "messages": messages,
                "temperature": 0.5,
                "stream": True,
            }
            try:
                response = requests.post(or_url, headers=headers_or, json=payload_or, stream=True, timeout=60)
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                pass
            except Exception as e:
                logger.error("OpenRouter fallback failed: %s", e)
                yield "Both AI providers are currently unavailable."

# --- 4. FLASK ROUTES ---

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "Velkor AI Backend"}), 200

@app.route("/api/chat/stream", methods=["POST", "OPTIONS"])
@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat_route():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data = request.form if request.form else request.get_json(silent=True) or {}
        messages_raw = data.get("messages", "[]")
        messages = json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
        research_mode = str(data.get("research_mode", "false")).lower() == "true"

        file_context = ""
        if "files" in request.files:
            uploaded_files = request.files.getlist("files")[:10]
            parsed_parts = []
            for f in uploaded_files:
                if not f or not f.filename:
                    continue
                text = DocumentService.parse_file(f)
                parsed_parts.append(f"--- FILE: {f.filename} ---\n{text}")
            file_context = "\n\n".join(parsed_parts)

        web_context = ""
        research_sources = []
        if research_mode and messages:
            latest_query = messages[-1].get("content", "").strip()
            if latest_query:
                research_data = SearchService.research(latest_query, search_count=8, pages_to_read=5)
                if research_data.get("success"):
                    web_context = research_data.get("context", "")
                    research_sources = research_data.get("sources", [])

        current_date = datetime.now().strftime("%A, %B %d, %Y")
        system_prompt = f"You are Velkor AI, a helpful assistant created by Gururaj Achar. Today's date is {current_date}. if user asks for the developer contact hten only you can say that 'you can contact Gururaj Achar by clicking the link at the bottom of the side bar', and also dont over expalin anything just simple and small also not very short expilain what is needed and user hasked for , keep the answer short and easy to understand and informative"

        if web_context:
            system_prompt += f"\n\nLIVE WEB RESEARCH:\n{web_context}"
        if file_context:
            system_prompt += f"\n\nAttached Document Content:\n{file_context}"

        final_messages = [{"role": "system", "content": system_prompt}] + messages

        return Response(
            stream_with_context(AIProviderRouter.generate_stream(final_messages)),
            content_type="text/event-stream"
        )
    except Exception as e:
        logger.exception("Chat route failed")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/portfolio/chat", methods=["POST", "OPTIONS"])
def portfolio_chat_route():
    """
    Dedicated route for the floating assistant widget on
    gururajachar2008.github.io. Kept separate from /api/chat so the
    portfolio's system prompt, context size limits, and future
    changes (e.g. rate limiting) never affect the classroom app.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.get_json(silent=True) or {}
        messages = data.get("messages", [])
        portfolio_context = str(data.get("context", ""))[:MAX_CONTEXT_CHARS]

        if not isinstance(messages, list) or not messages:
            return jsonify({"success": False, "error": "messages is required"}), 400

        # Keep only the last N turns and only the fields we expect.
        safe_messages = []
        for m in messages[-MAX_PORTFOLIO_MESSAGES:]:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str):
                safe_messages.append({"role": role, "content": content[:4000]})

        if not safe_messages:
            return jsonify({"success": False, "error": "no valid messages"}), 400

        current_date = datetime.now().strftime("%A, %B %d, %Y")
        system_prompt = (
            "You are Velkor, the AI assistant embedded in Gururaj Achar's "
            "personal portfolio site. Today's date is " + current_date + ". "
            "Answer visitor questions about Gururaj's skills, projects, and "
            "background using the PORTFOLIO CONTEXT below. Keep replies short "
            "(2-4 sentences) and friendly. If asked for contact info, say they "
            "can reach Gururaj through the contact section of the site. If a "
            "question isn't covered by the context, say you're not sure and "
            "suggest checking the Projects or Contact section."
        )

        if portfolio_context:
            system_prompt += f"\n\nPORTFOLIO CONTEXT:\n{portfolio_context}"

        final_messages = [{"role": "system", "content": system_prompt}] + safe_messages

        return Response(
            stream_with_context(AIProviderRouter.generate_stream(final_messages)),
            content_type="text/event-stream"
        )
    except Exception as e:
        logger.exception("Portfolio chat route failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/image", methods=["POST", "OPTIONS"])
def image_route():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"success": False, "error": "Prompt is required."}), 400

    try:
        from urllib.parse import quote
        encoded = quote(prompt)
        img_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
        r = requests.get(img_url, timeout=70)
        if r.status_code == 200:
            b64_data = base64.b64encode(r.content).decode("ascii")
            return jsonify({"success": True, "image_b64": b64_data, "mime": "image/png"})
        return jsonify({"success": False, "error": "Image generation provider failed."}), 502
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
