# -*- coding: utf-8 -*-
from fastapi import FastAPI, Body, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import re, html, os
from difflib import SequenceMatcher
from collections import defaultdict

app = FastAPI(title="News Dedup & TopK API", version="1.4.0")

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

# =============================================================================
# 구성 파라미터 (필요 시 환경변수로 조정)
# =============================================================================
# 기본값을 한국어 제목 특성에 맞게 다소 완화했습니다. (운영 환경에서 자유롭게 조정)
SIM_RATIO   = float(os.getenv("SIM_RATIO",   "0.78"))  # 기존 0.82 -> 0.78
JACCARD     = float(os.getenv("JACCARD",     "0.50"))  # 기존 0.60 -> 0.50

# 보조 규칙(문자 n-그램 / 오버랩 계수)
CHAR_JACC   = float(os.getenv("CHAR_JACC",   "0.45"))
CHAR_N      = int(os.getenv("CHAR_N",       "3"))
OVERLAP     = float(os.getenv("OVERLAP",     "0.70"))

TOPK        = int(os.getenv("TOPK",          "3"))

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

# HTML 태그 제거
TAG_RE = re.compile(r"<[^>]+>")
# 한글/영문 경계 판단용
KO = r"[가-힣]"
EN = r"[A-Za-z0-9]"

# -----------------------------------------------------------------------------
# 정규화 (강화)
# -----------------------------------------------------------------------------
def norm(s: Optional[str]) -> str:
    """
    제목/본문 정규화:
    - HTML 태그 제거
    - [속보], [단독], [종합] 등 노이즈 토큰 제거
    - 한글-영문/숫자 경계에 공백 삽입 -> 토큰경계 안정화
    - 주요 변형어 정규화 (sk networks -> sk네트웍스, 기분 금리 -> 기준금리 등)
    - 소문자화 + 특수문자 제거 + 공백 정리
    """
    s = html.unescape(s or "")
    s = TAG_RE.sub(" ", s)

    # 대괄호 라벨 제거: [속보], [단독], [종합], [포토], [영상], [특징주], ...
    s = re.sub(r"\[(?:\s*(속보|단독|종합|포토|영상|특징주|인터뷰|기획|르포|오피니언|사설)\s*)\]", " ", s, flags=re.IGNORECASE)
    # 라벨 단어 단독 등장도 제거(선택)
    s = re.sub(r"\b(속보|단독|종합)\b", " ", s, flags=re.IGNORECASE)

    # 한글-영문/숫자 경계에 공백 삽입 (예: "sk네트웍스" -> "sk 네트웍스")
    s = re.sub(f"({EN})({KO})", r"\1 \2", s)
    s = re.sub(f"({KO})({EN})", r"\1 \2", s)

    # 표기 통일(도메인 사전)
    repl = [
        (r"\bsk\s*[- ]?\s*networks\b", "sk네트웍스"),
        (r"\bsk\s*네트웍스\b", "sk네트웍스"),
        (r"\bboe\b", "영국중앙은행"),
        (r"기준\s*금리", "기준금리"),
        (r"초거대\s*언어\s*모델", "초거대언어모델"),
        (r"\bllm\b", "llm"),
    ]
    for pat, rep in repl:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    # 일반 치환
    s = s.replace("…", " ").replace("’", "'").replace("”", '"').replace("“", '"').replace("·", " ")
    s = s.lower()
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------------------------------------------------------------
# 유사도/스코어
# -----------------------------------------------------------------------------
def keyword_hits(text: str) -> int:
    t = text.casefold()
    total = 0
    for kw in KWS:
        # 영문/숫자 단어는 단어경계, 한글/혼합은 서브스트링 매칭
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

def char_jaccard(a: str, b: str, n: int = CHAR_N) -> float:
    if n <= 0:
        return 0.0
    sa = {a[i:i+n] for i in range(len(a)-n+1)} if len(a) >= n else set()
    sb = {b[i:i+n] for i in range(len(b)-n+1)} if len(b) >= n else set()
    return (len(sa & sb) / len(sa | sb)) if sa and sb else 0.0

def overlap_coeff(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return (len(sa & sb) / min(len(sa), len(sb))) if sa and sb else 0.0

# -----------------------------------------------------------------------------
# 원문 fetch (선택)
# -----------------------------------------------------------------------------
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

# --------- 유연 파서(확장) ---------
def extract_items(payload: Any) -> List[Dict[str, Any]]:
    """
    허용하는 모든 형태에서 NewsItem dict 배열을 뽑아낸다.
      - { "items": [ {...}, ... ] }
      - { "array": [ {...}, ... ] }
      - { "results": [ {...}, ... ] } / { "data": [...] }
      - [ {...}, {...} ]                             # 아이템 배열 자체
      - [ { "items": [ ... ] }, { "items": [ ... ] } ]
      - [ { "array": [ ... ] }, { "array": [ ... ] } ]
      - 메타 래퍼 { "lastBuildDate": "...", "items": [ ... ] }
    """
    candidate_keys = ("items", "array", "results", "data")

    def looks_like_items(lst: Any) -> bool:
        return (
            isinstance(lst, list) and
            len(lst) > 0 and
            all(isinstance(x, dict) and "title" in x for x in lst)
        )

    if isinstance(payload, dict):
        for k in candidate_keys:
            v = payload.get(k)
            if looks_like_items(v):
                return v
        if any(k in payload for k in ("lastBuildDate", "display", "total")):
            for k in candidate_keys:
                v = payload.get(k)
                if looks_like_items(v):
                    return v

    if isinstance(payload, list):
        if looks_like_items(payload):
            return payload
        merged: List[Dict[str, Any]] = []
        for el in payload:
            if isinstance(el, dict):
                for k in candidate_keys:
                    v = el.get(k)
                    if looks_like_items(v):
                        merged.extend(v)
        if merged:
            return merged

    raise HTTPException(
        status_code=400,
        detail=(
            "Invalid payload. Provide {items:[...]} or {array:[...]} "
            "or [ {items:[...]}, ... ] / [ {array:[...]}, ... ] / a plain array of items."
        )
    )

# --------- 헬스체크 ---------
@app.get("/ping")
def ping():
    return {"status": "ok"}

# --------- 랭킹(+중복제거) ---------
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

    # 정규화
    ntitles = [norm(it.title) for it in items]
    ndescs  = [norm(it.description or "") for it in items]
    fulls   = [f"{ntitles[i]} {ndescs[i]}".strip() for i in range(n)]
    kwscore = [keyword_hits(fulls[i]) for i in range(n)]

    # Disjoint-set (Union-Find)
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 중복 판정 규칙 (보수→완화 순서로 판정)
    for i in range(n):
        ti = ntitles[i]
        for j in range(i + 1, n):
            tj = ntitles[j]
            if not ti or not tj:
                continue

            # 1) 포함 관계
            if ti in tj or tj in ti:
                union(i, j); continue

            # 2) SequenceMatcher
            if title_sim(ti, tj) >= SIM_RATIO:
                union(i, j); continue

            # 3) 단어 Jaccard
            if jaccard(ti, tj) >= JACCARD:
                union(i, j); continue

            # 4) 문자 n-그램 Jaccard (보조)
            if char_jaccard(ti, tj, CHAR_N) >= CHAR_JACC:
                union(i, j); continue

            # 5) Overlap 계수 (보조)
            if overlap_coeff(ti, tj) >= OVERLAP:
                union(i, j); continue

    # 군집 구성
    clusters = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)

    # 대표 선택: 키워드 점수, 설명 길이, 인덱스 역순(안정성) 우선
    def choose_rep(group: List[int]) -> int:
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

    # 필요 시 원문 fetch하여 대표의 키워드 점수 재계산(랭킹 안정화용)
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

    # 최종 정렬: 중복크기 ↓, 키워드점수 ↓, 대표제목(정규화) ↑
    groups.sort(key=lambda g: (g["dup_count"], g["rep_kw"], ntitles[g["rep"]]), reverse=True)

    take = top_k if top_k is not None else max(1, TOPK)
    top_groups = groups[:max(1, take)]

    return [items[g["rep"]] for g in top_groups]
