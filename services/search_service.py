"""
services/search_service.py

Web search wrapper for Velkor AI.

Primary:  Serper (google.serper.dev) when SERPER_API_KEY is configured.
Fallback: DuckDuckGo HTML (keyless) so research mode keeps working without a
          paid key.
"""

import re

import requests
from config import Config
from utils.logger import get_logger
from utils.retry import retry

logger = get_logger(__name__)


class SearchService:
    URL = "https://google.serper.dev/search"
    DDG_URL = "https://html.duckduckgo.com/html/"

    def __init__(self):
        self.api_key = Config.SERPER_API_KEY

    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, num: int = 5):
        if self.api_key:
            result = self._search_serper(query, num)
            if result.get("success"):
                return result
            logger.warning(
                "Serper search failed (%s); falling back to DuckDuckGo.",
                result.get("error", "unknown"),
            )
        else:
            logger.info("SERPER_API_KEY missing; using keyless DuckDuckGo search.")

        return self._search_duckduckgo(query, num)

    @retry(max_retries=2)
    def _search_serper(self, query: str, num: int = 5):
        try:
            r = requests.post(
                self.URL,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": num},
                timeout=Config.REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                return {"success": False, "results": [], "error": r.text, "status_code": r.status_code}
            data = r.json()
            results = []
            for item in data.get("organic", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                })
            logger.info("Serper search '%s' returned %d results", query, len(results))
            return {"success": True, "results": results, "status_code": 200}
        except Exception as e:
            logger.exception("Serper search failed")
            return {"success": False, "results": [], "error": str(e), "status_code": 500}

    def _search_duckduckgo(self, query: str, num: int = 5):
        try:
            r = requests.post(
                self.DDG_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/124.0 Safari/537.36"
                },
                data={"q": query},
                timeout=Config.REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                return {"success": False, "results": [], "error": f"HTTP {r.status_code}", "status_code": r.status_code}

            results = self._parse_duckduckgo(r.text, num)
            logger.info("DuckDuckGo search '%s' returned %d results", query, len(results))
            return {"success": True, "results": results, "status_code": 200}
        except Exception as e:
            logger.exception("DuckDuckGo search failed")
            return {"success": False, "results": [], "error": str(e), "status_code": 500}

    @staticmethod
    def _parse_duckduckgo(html: str, num: int = 5):
        results = []
        # Each result block: <a class="result__a" href="...">title</a>
        # ... <a class="result__snippet" ...>snippet</a>
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.DOTALL,
        )
        snippets = re.findall(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html,
            flags=re.DOTALL,
        )

        for i, (href, title) in enumerate(blocks[:num]):
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i])
                snippet = re.sub(r"\s+", " ", snippet).strip()

            title = re.sub(r"<[^>]+>", "", title).strip()
            url = href.strip()
            # DuckDuckGo wraps redirects in uddg= parameters.
            m = re.search(r"uddg=([^&]+)", url)
            if m:
                try:
                    from urllib.parse import unquote
                    url = unquote(m.group(1))
                except Exception:
                    pass

            if not title and not url:
                continue
            results.append({"title": title, "snippet": snippet, "url": url})

        return results


search_service = SearchService()
