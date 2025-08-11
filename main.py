# -*- coding: utf-8 -*-
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import re, html, os
from difflib import SequenceMatcher
from collections import defaultdict

app = FastAPI(title="News Dedup & TopK API", version="1.2.1")

# ---- CORS (Make, 브라우저 호출 대비) ----
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

# ---- 구성 파라미터(환경변수로 조절 가능) ----
SIM_RATIO   = float(os.getenv("SIM_RATIO",  "0.82"))  # 제목 시퀀스 유사도 임계
JACCARD     = float(os.getenv("JACCARD",    "0.60"))  # 제목 자카드 임계
TOPK        = int(os.getenv("TOPK",         "3"))     # ✅ 기본 상위 3개만 반환

# ---- (링크 본문 lazy fetch: 점수 개선 전용) ----
FETCH_BY_LINK = os.getenv("FETCH_BY_LINK", "true").lower() == "true"  # 대표 기사만 링크 본문 시도
MAX_FETCH     = int(os.getenv("MAX_FETCH", "12"))       # 요청당 최대 fetch 수(대표 기준)
HTTP_TIMEOUT  = float(os.getenv("HTTP_TIMEOUT", "5.0"))
MAX_BYTES     = int(os.getenv("MAX_BYTES", "800000"))   # HTML 최대 바이트
USER_AGENT    = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; NewsDedupBot/1.0)")

# ---- 키워드 ----
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
            pat = r"\b" + re.escape(kw) + r"\b"   # 영문/스페이스 키워드 → 단어 경계
        else:
            pat = re.escape(kw)                    # 한글 키워드 → 포함 매치
        total += len(re.findall(pat, t))
    return total

def title_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return (len(sa & sb) / len(sa | sb)) if sa and sb else 0.0

# --------- (선택) 링크 본문 가져오기 ---------
def fetch_article_text(url: Optional[str]) -> Optional[str]:
    """가벼운 HTML 텍스트 추출. 실패/부적합이면 None."""
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

# --------- Pydantic 모델 ---------
class NewsItem(BaseModel):
    title: str
    description: Optional[str] = ""
    link: Optional[str] = None
    originallink: Optional[str] = None
    pubDate: Optional[str] = None

class InputPayload(BaseModel):
    items: List[NewsItem] = Field(..., description="뉴스 아이템 배열(권장 50개)")

# --------- 헬스체크 ---------
@app.get("/ping")
def ping():
    return {"status": "ok"}

# --------- 핵심 엔드포인트 ---------
# 응답은 입력과 동일 스키마(NewsItem)로 반환
@app.post("/rank", response_model=List[NewsItem])
def rank(payload: Any = Body(...)):
    """
    허용 형식:
    1) { "items": [ {title, description, link?, originallink?, pubDate?}, ... ] }
    2) [ {title, description, link?, originallink?, pubDate?}, ... ]
    """
    # 유연 파싱
    if isinstance(payload, dict) and "items" in payload:
        items_raw = payload["items"]
    elif isinstance(payload, list):
        items_raw = payload
    else:
        raise HTTPException(status_code=400, detail="Invalid payload. Provide {items:[...]} or an array of items.")

    items = [NewsItem(**it) for it in items_raw]
    n = len(items)
    if n == 0:
        return []

    # 1) 전처리 (제목+description 기반 1차 키워드 점수)
    ntitles = [norm(it.title) for it in items]
    ndescs  = [norm(it.description or "") for it in items]
    fulls   = [f"{ntitles[i]} {ndescs[i]}".strip() for i in range(n)]
    kwscore = [keyword_hits(fulls[i]) for i in range(n)]

    # 2) 제목 기반 중복 판단 → Union-Find
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

    # 3) 클러스터 대표 선택
    def choose_rep(group):
        # 1) 키워드 발생 수 최대, 2) description 길이, 3) 인덱스 작은 것
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

    # 4) (Lazy fetch) 대표 기사 링크 본문으로 rep_kw 보강(응답 내용은 원본 유지)
    if FETCH_BY_LINK and MAX_FETCH > 0:
        fetched = 0
        # fetch 우선순위
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

    # 5) 최종 정렬: 1) 중복수 desc 2) 키워드수 desc 3) 제목
    groups.sort(key=lambda g: (g["dup_count"], g["rep_kw"], ntitles[g["rep"]]), reverse=True)
    top_groups = groups[:max(1, TOPK)]

    # 6) 응답: 대표 기사를 "원본 그대로" 반환 (필드 가공/추가 없음)
    return [items[g["rep"]] for g in top_groups]
