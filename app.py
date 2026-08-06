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
from datetime import datetime
import traceback

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
CORS(app, resources={r"/api/*": {"origins": ["https://gururajachar2008.github.io/Velkor", "http://localhost:5173", "http://localhost:3000"]}}, supports_credentials=True)

# --- CONFIGURATION ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not NVIDIA_API_KEY:
    logger.warning("NVIDIA_API_KEY is not set — NVIDIA NIM calls will be skipped.")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY is not set — the OpenRouter fallback will fail.")

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
                timeout=9,
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

# --- 2. DOCUMENT / FILE PARSING SERVICE (Universal Support) ---
class DocumentService:
    @staticmethod
    def parse_file(file_storage) -> str:
        """Extracts text seamlessly from code files, PDFs, Word, Excel, PowerPoint, and text formats."""
        filename = file_storage.filename or "upload.bin"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file_storage.save(path)

        extracted_text = ""
        ext = filename.split(".")[-1].lower()
        try:
            # Plain text, source code, and data formats
            if ext in ["txt", "py", "js", "ts", "jsx", "tsx", "html", "css", "json", "md", "csv", "xml", "yaml", "yml", "sql", "sh"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
            
            # PDF documents
            elif ext == "pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(path)
                    extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                except Exception:
                    extracted_text = "[PDF parsing library unavailable, binary stored]"
            
            # Word documents (.docx)
            elif ext == "docx":
                try:
                    from docx import Document
                    doc = Document(path)
                    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                    tables_text = []
                    for table in doc.tables:
                        for row in table.rows:
                            row_vals = [cell.text.strip() for cell in row.cells]
                            tables_text.append(" | ".join(row_vals))
                    extracted_text = "\n".join(paragraphs)
                    if tables_text:
                        extracted_text += "\n\nTables Content:\n" + "\n".join(tables_text)
                except Exception as e:
                    logger.error("DOCX parse error: %s", e)
                    extracted_text = "[Error parsing Word document (.docx)]"

            # Excel spreadsheets (.xlsx)
            elif ext in ["xlsx", "xls"]:
                try:
                    import pandas as pd
                    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl" if ext == "xlsx" else "xlrd")
                    sheet_dumps = []
                    for s_name, df in sheets.items():
                        sheet_dumps.append(f"--- Sheet: {s_name} ---\n" + df.to_string(index=False))
                    extracted_text = "\n\n".join(sheet_dumps)
                except Exception as e:
                    logger.error("Excel parse error: %s", e)
                    extracted_text = "[Error parsing Spreadsheet]"

            # PowerPoint presentations (.pptx)
            elif ext == "pptx":
                try:
                    from pptx import Presentation
                    prs = Presentation(path)
                    slide_texts = []
                    for idx, slide in enumerate(prs.slides, start=1):
                        slide_lines = [f"Slide {idx}:"]
                        for shape in slide.shapes:
                            if shape.has_text_frame:
                                for para in shape.text_frame.paragraphs:
                                    t = "".join(run.text for run in para.runs).strip()
                                    if t:
                                        slide_lines.append(t)
                        slide_texts.append("\n".join(slide_lines))
                    extracted_text = "\n\n".join(slide_texts)
                except Exception as e:
                    logger.error("PPTX parse error: %s", e)
                    extracted_text = "[Error parsing Presentation (.pptx)]"

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

        nvidia_error = None
        success = False
        if NVIDIA_API_KEY:
            try:
                start_time = time.time()
                response = requests.post(nvidia_url, headers=headers_nvidia, json=payload_nvidia, stream=True, timeout=10)
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
                else:
                    nvidia_error = f"NVIDIA NIM returned HTTP {response.status_code}"
            except Exception as e:
                nvidia_error = f"NVIDIA NIM failed: {e}"

        # Fallback: OpenRouter Free Tier
        if not success:
            if not OPENROUTER_API_KEY:
                yield "I can't reach the AI providers. Please configure your API keys."
                return

            or_url = "https://openrouter.ai/api/v1/chat/completions"
            headers_or = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://velkor.ai",
                "X-Title": "Velkor AI",
            }
            payload_or = {
                "model": "deepseek/deepseek-r1-0528:free",
                "messages": messages,
                "temperature": 0.5,
                "stream": True,
            }
            try:
                response = requests.post(or_url, headers=headers_or, json=payload_or, stream=True, timeout=15)
                response.raise_for_status()
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
                logger.error("OpenRouter fallback failed: %s", e)
                yield "Both AI providers are currently unavailable."

# --- 4. FLASK ROUTES ---

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "Velkor AI Backend"}), 200

@app.route("/api/chat/stream", methods=["POST", "OPTIONS"])
def chat_stream_route():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    return chat_route()

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat_route():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.form if request.form else request.get_json(silent=True) or {}
        messages_raw = data.get("messages", "[]")
        import json
        messages = json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
        research_mode = str(data.get("research_mode", "false")).lower() == "true"

        file_context = ""
        if "file" in request.files:
            file_obj = request.files["file"]
            file_context = DocumentService.parse_file(file_obj)

        web_context = ""
        if research_mode and messages:
            latest_query = messages[-1].get("content", "")
            search_res = SearchService.search(latest_query)
            if search_res.get("success"):
                web_context = "\n".join([f"- {r['title']}: {r['snippet']} ({r['url']})" for r in search_res["results"]])

        current_date = datetime.now().strftime("%A, %B %d, %Y")
        system_prompt = f"""You are Velkor AI, a helpful and thoughtful assistant created by Gururaj Achar.

Today's date is {current_date}. Your training data has a fixed cutoff in the past, but that cutoff is not "now" — treat any event dated at or before today as real, even if it happened after your training cutoff. Never describe a dated fact, election result, appointment, or news item as "speculative," "forward-dated," or "not yet real" just because you don't personally recall it — that reaction means your information is outdated, not that the fact is fake.

Respond warmly and naturally, like a knowledgeable friend having a conversation — not a lecturer. Match your response length to the question: quick questions get quick answers, complex ones get the space they need. Don't pad responses with unnecessary structure, headers, or step-by-step breakdowns unless the user asks for them or the content genuinely calls for it (e.g. instructions, code, comparisons).

Use clean Markdown formatting where it helps readability. Use fenced code blocks only for actual code — keep regular explanations as plain text, not bullet-crammed or over-formatted.

Respond in English by default. Respond in another language only if the user writes in or requests that language.

You have access to live web search for current information — news, prices, recent events, anything time-sensitive. When your answer depends on something that could have changed recently, use it rather than guessing or hedging about your knowledge cutoff. Don't tell the user you "can't access the internet" or "don't have real-time data" — you do.

When a user uploads a file (PDF, document, image, etc.), the backend extracts and passes you its full content directly in the conversation. Treat that content as something you can already see in full — don't ask the user to paste or describe it, and don't claim you can't read attachments.

Be direct and honest. If you don't know something or a search doesn't turn up a clear answer, say so plainly instead of filling the gap with a confident-sounding guess.

If asked how to contact your developer, respond with: "You can reach out to Gururaj Achar at https://gururajachar2008.github.io/Portfolio2.0/". Only share this when asked directly — never volunteer it."""

        if web_context:
            system_prompt += f"""
            Live search results, retrieved just now on {current_date}:
            {web_context}
            Treat these as your primary source for anything time-sensitive — current facts, prices, 
            specifications, comparisons, recent announcements, office-holders, and live events. Prioritize 
            them over your own prior knowledge whenever the two could conflict, since search results reflect 
            the current state of things and your training data may not. Take the results at face value: if a 
            result's date is at or before today, it is a real, already-happened event, not a prediction or 
            speculation — do not editorialize about a date 'seeming premature' or 'looking forward-dated.' 
            If the results don't fully answer the question, say what's missing rather than filling the gap 
            from memory. If results genuinely conflict with each other (not just with what you expected), 
            note the discrepancy instead of picking one silently.
            {web_context}
            """
        if file_context:
            system_prompt += f"\n\nAttached Document Content:\n{file_context}"

        final_messages = [{"role": "system", "content": system_prompt}] + messages

        return Response(
            stream_with_context(AIProviderRouter.generate_stream(final_messages)),
            content_type="text/event-stream"
        )
    except Exception as e:
        traceback.print_exc()
        logger.exception("Chat route failed")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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
        traceback.print_exc()
        logger.exception("Image generation failed")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
