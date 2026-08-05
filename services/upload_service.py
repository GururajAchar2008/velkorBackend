"""
services/upload_service.py
File upload parsing orchestration.
"""

import os
import uuid
from typing import Dict, Any

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import Config
from files.extractor import extract_file, extract_file_path
from utils.logger import get_logger, log_upload

logger = get_logger(__name__)


class UploadService:
    def process(self, file: FileStorage) -> Dict[str, Any]:
        if not file or not file.filename:
            return {"success": False, "error": "No file provided."}

        filename = secure_filename(file.filename) or "upload.bin"
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > Config.MAX_UPLOAD_BYTES:
            return {
                "success": False,
                "error": f"File too large. Maximum size is {Config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                "status_code": 413,
            }

        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        unique = f"{uuid.uuid4().hex[:10]}_{filename}"
        path = os.path.join(Config.UPLOAD_FOLDER, unique)

        try:
            file.save(path)
            log_upload(filename, size)
            extracted = extract_file_path(path)
            if not extracted.get("success"):
                return {
                    "success": False,
                    "error": extracted.get("error") or "Failed to parse file.",
                    "filename": filename,
                    "size": size,
                }

            text = extracted.get("text") or ""
            return {
                "success": True,
                "file_id": unique,
                "filename": filename,
                "size": size,
                "text": text,
                "metadata": extracted.get("metadata") or {},
                "pages": extracted.get("pages") or 0,
                "preview": text[:500],
            }
        except Exception as e:
            logger.error("Upload failed for %s: %s", filename, e)
            return {"success": False, "error": str(e), "filename": filename}
        finally:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


upload_service = UploadService()
