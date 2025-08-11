# -*- coding: utf-8 -*-
from fastapi import FastAPI, Body, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import re, html, os
from difflib import SequenceMatcher
from collections import defaultdict

app = FastAPI(title="News Dedup & TopK API", version="1.3.0")

# ---- CORS ----
try:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

# ---- 구성 파라미터 ----
SIM_RATIO   = float(os.getenv("SIM_RATIO",  "0.82"))
JACCARD     = float(os.getenv("JACCARD",    "0.60"))
TOPK        = int(os.getenv("TOPK",         "3"))

FETCH_BY_LINK = os.getenv("FETCH_BY_LINK", "true").lower() == "true"
MAX_FETCH     = int(os.getenv("MAX_FETCH", "12"))
HTTP_TIMEOUT  = float(os.getenv("HTTP_TIMEOUT", "5.0"))
MAX_BYTES     = int(os.getenv("MAX_BYTES", "800000"))
USER_AGENT    = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; NewsDedupBot/1.0)")

RAW_KWS = [
    'ai', '인공지능', '로보틱스', '미래산업', '자율주행',
    'sk네트웍스', 'sk networks', 'sk매직', '피닉스랩', 'LLM',
    '생성형 AI', 'Lama', '그록', '일론 머스크', 'sk', 'lg',
    '샘 올트먼', '민팃', '생성형 ai', '삼성'
]
KWS = [k.casefold() for k in RAW_KWS]
TAG_RE = re.compile(r"<[^>]+>")

def norm(s: Optional[str]) -> str:
    s = html.unescape(s or "")
    s = TAG_RE.sub(" ", s)
    s = s.replace("…"," ").replace("’","'").replace("”",'"').replace("“",'"')
    s = s.lower()
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def keyword_hits(text: str) -> int:
    t = text.casefold()
    total = 0
    for kw in KWS:
        if re.fullmatch(r"[0-9a-z]+(\s[0-9a-z]+)*", kw):
            pat = r"\b" + re.escape(kw) + r"\b"
        else:
            pat = re.escape(kw)
        total += len(re.findall(pat, t))
    return total

def title_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return (len(sa & sb) / len(sa | sb)) if sa and sb else 0.0

def fetch_article_text(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        return None

    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT,
                            headers={"User-Agent": USER_AGENT}, stream=True)
        resp.raise_for_status()
        content = b""
        for chunk in resp.iter_content(65536):
            content += chunk
            if len(content) > MAX_BYTES:
                break
        html_text = content.decode(resp.encoding or "utf-8", errors="ignore")
    except Exception:
        return None

    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        candidates = []
        for sel in ["article", "main", "[role=main]", ".newsct_article", "#dic_area",
                    ".article_body", "#articeBody", "#newsEndContents"]:
            c = soup.select_one(sel)
            if c:
                txt = re.sub(r"\s+", " ", c.get_text(separator=" ")).strip()
                if len(txt) >= 50:
                    candidates.append(txt)
        if candidates:
            text = max(candidates, key=len)
        else:
            text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()

        return text if len(text) >= 50 else None
    except Exception:
        return None

# --------- 모델 ---------
class NewsItem(BaseModel):
    title: str
    description: Optional[str] = ""
    link: Optional[str] = None
    originallink: Optional[str] = None
    pubDate: Optional[str] = None

# --------- 유연 파서(핵심 수정) ---------
def extract_items(payload: Any) -> List[Dict[str, Any]]:
    """
    허용하는 모든 형태에서 NewsItem dict 배열을 뽑아낸다.
    - { "items": [ {...}, ... ] }
    - [ {...}, {...} ]  # 뉴스 아이템 배열 자체
    - [ { "items": [ {...}, ... ], ... }, { "items": [ ... ] } ]  # ✅ 이번 케이스(네이버/검색 래퍼)
    - 단일 래퍼 { "lastBuildDate": "...", "items": [ ... ] }
    """
    # dict + items
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"]

    # 최상위가 배열
    if isinstance(payload, list):
        # 배열이 곧바로 뉴스 아이템인지 감지
        if payload and isinstance(payload[0], dict) and "title" in payload[0]:
            return payload  # 이미 NewsItem 배열

        # 배열 안에 래퍼들이 있고, 각 래퍼에 items 배열이 있는 경우 → 전부 합치기
        merged: List[Dict[str, Any]] = []
        for el in payload:
            if isinstance(el, dict) and isinstance(el.get("items"), list):
                merged.extend(el["items"])
        if merged:
            return merged

    # 단일 래퍼 (배열이 아니지만 lastBuildDate 등의 메타만 있고 items가 있는 경우)
    if isinstance(payload, dict) and any(k in payload for k in ("lastBuildDate", "display", "total")):
        if isinstance(payload.get("items"), list):
            return payload["items"]

    raise HTTPException(
        status_code=400,
        detail="Invalid payload. Provide {items:[...]} OR an array of items OR an array of wrappers each having items."
    )

# --------- 헬스체크 ---------
@app.get("/ping")
def ping():
    return {"status": "ok"}

# --------- 랭킹 ---------
@app.post("/rank", response_model=List[NewsItem])
def rank(
    payload: Any = Body(...),
    top_k: int = Query(None, ge=1, le=500, description="반환할 상위 개수(미지정 시 환경변수 TOPK 사용)")
):
    items_raw = extract_items(payload)
    items = [NewsItem(**it) for it in items_raw]
    n = len(items)
    if n == 0:
        return []

    ntitles = [norm(it.title) for it in items]
    ndescs  = [norm(it.description or "") for it in items]
    fulls   = [f"{ntitles[i]} {ndescs[i]}".strip() for i in range(n)]
    kwscore = [keyword_hits(fulls[i]) for i in range(n)]

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        ti = ntitles[i]
        for j in range(i + 1, n):
            tj = ntitles[j]
            if not ti or not tj:
                continue
            if ti in tj or tj in ti:
                union(i, j); continue
            if title_sim(ti, tj) >= SIM_RATIO:
                union(i, j); continue
            if jaccard(ti, tj) >= JACCARD:
                union(i, j); continue

    clusters = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)

    def choose_rep(group):
        best, key = None, None
        for i in group:
            sc = (kwscore[i], len(items[i].description or ""), -i)
            if key is None or sc > key:
                best, key = i, sc
        return best

    groups = []
    for _, idxs in clusters.items():
        rep = choose_rep(idxs)
        groups.append({"rep": rep, "dup_count": len(idxs), "rep_kw": kwscore[rep]})

    if FETCH_BY_LINK and MAX_FETCH > 0:
        fetched = 0
        groups.sort(key=lambda g: (g["dup_count"], g["rep_kw"], ntitles[g["rep"]]), reverse=True)
        for g in groups:
            if fetched >= MAX_FETCH:
                break
            rep = g["rep"]
            url = items[rep].link or items[rep].originallink
            if not url:
                continue
            txt = fetch_article_text(url)
            if not txt:
                continue
            fetched += 1
            full = f"{ntitles[rep]} {norm(txt)}".strip()
            g["rep_kw"] = keyword_hits(full)

    groups.sort(key=lambda g: (g["dup_count"], g["rep_kw"], ntitles[g["rep"]]), reverse=True)

    # 쿼리파라미터 우선, 없으면 환경변수 TOPK
    take = top_k if top_k is not None else max(1, TOPK)
    top_groups = groups[:max(1, take)]

    return [items[g["rep"]] for g in top_groups]
