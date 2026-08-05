"""
services/rag_service.py
Universal file → chunk → rank → context pipeline.
"""

from typing import Dict, List, Optional

from config import Config
from rag.chunking import chunk_text
from rag.ranking import rank_chunks
from rag.prompt_builder import build_document_context
from utils.logger import get_logger, log_rag

logger = get_logger(__name__)


class RAGService:
    def build_context(
        self,
        query: str,
        document_text: str,
        top_k: int = None,
    ) -> Dict:
        text = (document_text or "").strip()
        if not text:
            return {
                "context": "",
                "chunks_total": 0,
                "chunks_used": 0,
                "rag_used": False,
            }

        chunks = chunk_text(text)
        selected = rank_chunks(query or "", chunks, top_k=top_k or Config.TOP_K_CHUNKS)
        context = build_document_context(selected)
        log_rag("document", len(selected))
        return {
            "context": context,
            "chunks_total": len(chunks),
            "chunks_used": len(selected),
            "rag_used": bool(context),
        }


rag_service = RAGService()
