"""
app.py — Velkor AI Complete Backend (Single-File Architecture)
Includes: Chat (with streaming, NVIDIA primary -> OpenRouter fallback),
Research search, universal file upload/parsing, and image generation.
"""

import os
import time
import base64
import logging
from typing import Dict, Any, Generator
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("velkor-app")

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. SEARCH SERVICE ---
class SearchService:
    @staticmethod
    def search(query: str, num: int = 5) -> Dict[str, Any]:
        """Performs web search via DuckDuckGo HTML fallback for research mode."""
        url = "https://html.duckduckgo.com/html/"
        try:
            r = requests.post(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                data={"q": query},
                timeout=5,
            )
            if r.status_code != 200:
                return {"success": False, "results": []}
            
            import re, html
            blocks = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.DOTALL)
            snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
            
            results = []
            for i, (href, title) in enumerate(blocks[:num]):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]) if i < len(snippets) else ""
                clean_title = re.sub(r"<[^>]+>", "", title).strip()
                results.append({"title": clean_title, "snippet": html.unescape(snippet).strip(), "url": href})
            return {"success": True, "results": results}
        except Exception as e:
            logger.error("Search failed: %s", e)
            return {"success": False, "results": []}

# --- 2. DOCUMENT / FILE PARSING SERVICE ---
class DocumentService:
    @staticmethod
    def parse_file(file_storage) -> str:
        """Extracts text from uploaded code files, documents, and images."""
        filename = file_storage.filename or "upload.bin"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file_storage.save(path)
        
        extracted_text = ""
        ext = filename.split(".")[-1].lower()
        try:
            if ext in ["txt", "py", "js", "html", "css", "json", "md", "csv"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
            elif ext == "pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(path)
                    extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                except Exception:
                    extracted_text = "[PDF parsing library unavailable, binary stored]"
            else:
                extracted_text = f"[Uploaded file: {filename} of type {ext}]"
        except Exception as e:
            logger.error("File parse error: %s", e)
            extracted_text = f"[Error reading file {filename}]"
        finally:
            if os.path.exists(path):
                os.remove(path)
        return extracted_text

# --- 3. AI PROVIDER ROUTER (NVIDIA -> OpenRouter Fallback) ---
class AIProviderRouter:
    @staticmethod
    def generate_stream(messages: list) -> Generator[str, None, None]:
        # Primary Attempt: NVIDIA NIM (Nemotron 3 Ultra)
        nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers_nvidia = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        }
        payload_nvidia = {
            "model": "nvidia/nemotron-3-ultra-120b-a3b",  # Primary Model requested
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1024,
            "stream": True,
        }

        success = False
        if NVIDIA_API_KEY:
            try:
                start_time = time.time()
                response = requests.post(nvidia_url, headers=headers_nvidia, json=payload_nvidia, stream=True, timeout=3)
                if response.status_code == 200:
                    logger.info("NVIDIA NIM stream connected in %.2fs", time.time() - start_time)
                    success = True
                    for line in response.iter_lines():
                        if line:
                            decoded = line.decode("utf-8")
                            if decoded.startswith("data: "):
                                data_str = decoded[6:]
                                if data_str.strip() == "[DONE]":
                                    break
                                import json
                                try:
                                    chunk_json = json.loads(data_str)
                                    delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if delta:
                                        yield delta
                                except Exception:
                                    pass
                    return
            except Exception as e:
                logger.warning("NVIDIA NIM streaming failed or timed out: %s. Falling back to OpenRouter.", e)

        # Fallback: OpenRouter Free Tier
        if not success:
            logger.info("Switching to OpenRouter free tier fallback.")
            or_url = "https://openrouter.ai/api/v1/chat/completions"
            headers_or = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://velkor.ai",
                "X-Title": "Velkor AI",
            }
            payload_or = {
                "model": "deepseek/deepseek-chat:free", # Reliable fast free fallback model
                "messages": messages,
                "temperature": 0.5,
                "stream": True,
            }
            try:
                response = requests.post(or_url, headers=headers_or, json=payload_or, stream=True, timeout=5)
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            import json
                            try:
                                chunk_json = json.loads(data_str)
                                delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                pass
            except Exception as e:
                logger.error("OpenRouter fallback stream failed: %s", e)
                yield "I am currently experiencing high demand. Please try again in a moment."

# --- 4. FLASK ROUTES ---

@app.route("/api/chat", methods=["POST"])
def chat_route():
    """Chat endpoint supporting streaming, research mode, and file context attachments."""
    try:
        data = request.form if request.form else request.get_json(silent=True) or {}
        
        # Parse inputs
        messages_raw = data.get("messages", "[]")
        import json
        messages = json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
        research_mode = str(data.get("research_mode", "false")).lower() == "true"
        
        # Handle file upload context if attached
        file_context = ""
        if "file" in request.files:
            file_obj = request.files["file"]
            file_context = DocumentService.parse_file(file_obj)

        # Handle web search context if Research is enabled
        web_context = ""
        if research_mode and messages:
            latest_query = messages[-1].get("content", "")
            search_res = SearchService.search(latest_query)
            if search_res.get("success"):
                web_context = "\n".join([f"- {r['title']}: {r['snippet']} ({r['url']})" for r in search_res["results"]])

        # Construct final system instruction and prompt payload
        system_prompt = "You are Velkor AI, a helpful assistant created by Gururaj Achar."
        if web_context:
            system_prompt += f"\n\nLive Web Research Results:\n{web_context}"
        if file_context:
            system_prompt += f"\n\nAttached Document Content:\n{file_context}"

        final_messages = [{"role": "system", "content": system_prompt}] + messages

        return Response(
            stream_with_context(AIProviderRouter.generate_stream(final_messages)),
            content_type="text/event-stream"
        )
    except Exception as e:
        logger.error("Chat route error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/image", methods=["POST"])
def image_route():
    """Image generation route using Pollinations/NVIDIA endpoints."""
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"success": False, "error": "Prompt is required."}), 400
    
    try:
        from urllib.parse import quote
        encoded = quote(prompt)
        img_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
        r = requests.get(img_url, timeout=30)
        if r.status_code == 200:
            b64_data = base64.b64encode(r.content).decode("ascii")
            return jsonify({
                "success": True,
                "image_b64": b64_data,
                "mime": "image/png"
            })
        return jsonify({"success": False, "error": "Image generation provider failed."}), 502
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
