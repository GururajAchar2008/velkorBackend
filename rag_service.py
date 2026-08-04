"""
rag_service.py
Improved RAG service for Velkor AI.
"""

import os
import requests
from typing import List

SERPER_API_KEY=os.getenv("SERPER_API_KEY")

TIME_KEYWORDS=[
"latest","today","current","news","recent","price","weather",
"compare","comparison","vs","versus","2025","2026"
]

MAX_CONTEXT_CHARS=12000

def needs_web_search(query:str)->bool:
    if not query:
        return False
    q=query.lower()
    return any(k in q for k in TIME_KEYWORDS)

def chunk_text(text:str,chunk_size:int=1200,overlap:int=150)->List[str]:
    text=text.strip()
    if len(text)<=chunk_size:
        return [text]
    chunks=[]
    start=0
    while start<len(text):
        end=min(len(text),start+chunk_size)
        chunks.append(text[start:end])
        if end==len(text):
            break
        start=end-overlap
    return chunks

def rank_chunks(query:str,chunks:List[str],top_k:int=5)->List[str]:
    q=set(query.lower().split())
    scored=[]
    for c in chunks:
        score=sum(1 for w in q if w in c.lower())
        scored.append((score,c))
    scored.sort(reverse=True,key=lambda x:x[0])
    return [c for _,c in scored[:top_k]]

def web_search_context(query,max_results=5):
    if not SERPER_API_KEY:
        return ""
    try:
        r=requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY":SERPER_API_KEY,
                "Content-Type":"application/json"
            },
            json={"q":query,"num":max_results},
            timeout=15
        )
        r.raise_for_status()
        data=r.json()
        lines=[]
        for item in data.get("organic",[]):
            lines.append(f"{item.get('title','')}\n{item.get('snippet','')}\n{item.get('link','')}")
        return "\n\n".join(lines)
    except Exception:
        return ""

def build_prompt(base_prompt:str,query:str,file_context:str="")->str:
    prompt=base_prompt
    if needs_web_search(query):
        web=web_search_context(query)
        if web:
            prompt+="\n\nLive Search Results\n"+web
    if file_context:
        chunks=chunk_text(file_context)
        best=rank_chunks(query,chunks)
        prompt+="\n\nDocument Context\n"+"\n\n".join(best)[:MAX_CONTEXT_CHARS]
    return prompt
