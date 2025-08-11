# -*- coding: utf-8 -*-
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import re, html, textwrap, os
from difflib import SequenceMatcher
from collections import defaultdict

app = FastAPI(title="News Dedup & Top10 API", version="1.0.0")

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
SIM_RATIO = float(os.getenv("SIM_RATIO", "0.82"))   # 제목 시퀀스 유사도
JACCARD   = float(os.getenv("JACCARD",   "0.60"))   # 제목 자카드 임계치
DESC_WIDTH = int(os.getenv("DESC_WIDTH", "160"))    # 요약 길이

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

def clean_title(s: Optional[str]) -> str:
    return TAG_RE.sub("", html.unescape(s or "")).strip().lower()

def keyword_hits(text: str) -> int:
    t = text.casefold()
    total = 0
    for kw in KWS:
        if re.fullmatch(r"[0-9a-z]+(\s[0-9a-z]+)*", kw):
            pat = r"\b" + re.escape(kw) + r"\b"  # 영문/스페이스 포함 키워드
        else:
            pat = re.escape(kw)                  # 한글 키워드
        total += len(re.findall(pat, t))
    return total

def brief(desc: str, width: int = DESC_WIDTH) -> str:
    return textwrap.shorten(norm(desc), width=width, placeholder=" ...")

def title_sim(a: str, b: str) -> float:
    # SequenceMatcher는 50건 기준 1.2천회 호출 → 512MB에서도 충분
    return SequenceMatcher(None, a, b).ratio()

def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return (len(sa & sb) / len(sa | sb)) if sa and sb else 0.0

# --------- Pydantic 모델 ---------
class NewsItem(BaseModel):
    title: str
    description: Optional[str] = ""
    link: Optional[str] = None
    originallink: Optional[str] = None
    pubDate: Optional[str] = None

class InputPayload(BaseModel):
    # Notion/Make 단계에서 통일하기 쉬운 형태: items 배열
    items: List[NewsItem] = Field(..., description="뉴스 아이템 50개 배열")

class RankedItem(BaseModel):
    title: str
    description: str
    score: int
    count: int

# --------- 헬스체크 ---------
@app.get("/ping")
def ping():
    return {"status":"ok"}

# --------- 핵심 엔드포인트 ---------
@app.post("/rank", response_model=List[RankedItem])
def rank(payload: Any = Body(...)):
    """
    요청 바디는 두 형태를 모두 허용:
    1) { "items": [ {title, description, ...}, ... ] }  ← 권장
    2) [ {title, description, ...}, ... ]
    """
    # 유연 파싱
    items_raw: List[Dict[str, Any]]
    if isinstance(payload, dict) and "items" in payload:
        items_raw = payload["items"]
    elif isinstance(payload, list):
        items_raw = payload
    else:
        raise ValueError("Invalid payload. Provide {items: [...]} or an array of items.")

    items = [NewsItem(**it) for it in items_raw]
    n = len(items)
    if n == 0:
        return []

    # 전처리
    ntitles = [norm(it.title) for it in items]
    ndescs  = [norm(it.description or "") for it in items]
    fulls   = [f"{ntitles[i]} {ndescs[i]}".strip() for i in range(n)]
    kwscore = [keyword_hits(fulls[i]) for i in range(n)]

    # Union-Find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 중복 판단: 포함관계 or SIM_RATIO or JACCARD
    for i in range(n):
        ti = ntitles[i]
        for j in range(i+1, n):
            tj = ntitles[j]
            if not ti or not tj:
                continue
            if ti in tj or tj in ti:
                union(i,j); continue
            if title_sim(ti, tj) >= SIM_RATIO:
                union(i,j); continue
            if jaccard(ti, tj) >= JACCARD:
                union(i,j); continue

    clusters = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)

    def choose_rep(group):
        # 1) 키워드 발생 수 최대, 2) 설명 길이, 3) 인덱스 작은 것
        best, key = None, None
        for i in group:
            sc = (kwscore[i], len(items[i].description or ""), -i)
            if key is None or sc > key:
                best, key = i, sc
        return best

    groups = []
    for _, idxs in clusters.items():
        rep = choose_rep(idxs)
        groups.append({
            "rep": rep,
            "dup_count": len(idxs),
            "rep_kw": kwscore[rep],
        })

    # 정렬: (중복수 desc, 대표 키워드수 desc, 제목 사전순)
    groups.sort(
        key=lambda g: (g["dup_count"], g["rep_kw"], norm(items[g["rep"]].title)),
        reverse=True
    )
    top10 = groups[:10]

    results: List[RankedItem] = []
    for g in top10:
        rep = g["rep"]
        obj = RankedItem(
            title=f"{clean_title(items[rep].title)} ({g['dup_count']})",
            description=brief(items[rep].description or "", width=DESC_WIDTH),
            score=int(g["dup_count"] + g["rep_kw"]),
            count=int(g["dup_count"]),
        )
        results.append(obj)
    return results
