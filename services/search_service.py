"""
services/search_service.py
Serper search wrapper for Velkor AI.
"""

import requests
from config import Config
from utils.logger import get_logger
from utils.retry import retry

logger=get_logger(__name__)

class SearchService:
    URL="https://google.serper.dev/search"

    def __init__(self):
        self.api_key=Config.SERPER_API_KEY

    def available(self)->bool:
        return bool(self.api_key)

    @retry(max_retries=2)
    def search(self,query:str,num:int=5):
        if not self.api_key:
            return {"success":False,"results":[],"error":"SERPER_API_KEY missing","status_code":500}

        try:
            r=requests.post(
                self.URL,
                headers={
                    "X-API-KEY":self.api_key,
                    "Content-Type":"application/json"
                },
                json={"q":query,"num":num},
                timeout=Config.REQUEST_TIMEOUT
            )
            status=r.status_code
            if status!=200:
                return {"success":False,"results":[],"error":r.text,"status_code":status}
            data=r.json()
            results=[]
            for item in data.get("organic",[]):
                results.append({
                    "title":item.get("title",""),
                    "snippet":item.get("snippet",""),
                    "url":item.get("link","")
                })
            logger.info("Search '%s' returned %d results",query,len(results))
            return {"success":True,"results":results,"status_code":200}
        except Exception as e:
            logger.exception("Search failed")
            return {"success":False,"results":[],"error":str(e),"status_code":500}

search_service=SearchService()
