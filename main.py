from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from collections import defaultdict
import re

app = FastAPI()

# ✅ 데이터 구조 정의
class NewsItem(BaseModel):
    title: str
    description: str

class NewsRequest(BaseModel):
    items: List[NewsItem]

# ✅ 텍스트 정제 함수
def clean_text(s):
    s = re.sub('<.*?>', ' ', s or '')
    s = re.sub('[^0-9a-zA-Z가-힣 ]+', ' ', s)
    s = re.sub('\s+', ' ', s)
    return s.lower().strip()

# ✅ 공통 단어 수 계산
def common_words(a, b):
    set_a = set(a.split())
    set_b = set(b.split())
    return len(set_a & set_b)

# ✅ 키워드 리스트
keywords = [
    'ai', '인공지능', '로보틱스', '미래산업', '자율주행',
    'sk네트웍스', 'sk networks', 'sk매직', '피닉스랩', 'LLM',
    '생성형 AI', 'Lama', '그록', '일론 머스크', 'sk', 'lg',
    '샘 올트먼', '민팃', '생성형 ai', '삼성'
]

# ✅ 키워드 기반 스코어 계산
def keyword_score(title, desc, keywords):
    score = 0
    for kw in keywords:
        kw_l = kw.lower()
        score += 2 * title.count(kw_l)
        score += desc.count(kw_l)
    return score

# ✅ API 엔드포인트
@app.post("/deduplicate")
def deduplicate_news(request: NewsRequest):
    articles = []
    for item in request.items:
        title = clean_text(item.title)
        desc = clean_text(item.description)
        articles.append({'title': title, 'desc': desc})

    n = len(articles)
    group_ids = [-1] * n
    curr_gid = 0
    min_common = 3  # 공통 단어 3개 이상이면 동일 그룹

    # ✅ 기사 그룹화
    for i in range(n):
        if group_ids[i] == -1:
            group_ids[i] = curr_gid
            for j in range(i+1, n):
                if group_ids[j] == -1 and common_words(articles[i]['title'], articles[j]['title']) >= min_common:
                    group_ids[j] = curr_gid
            curr_gid += 1

    # ✅ 그룹별 대표 기사 선정
    group2indices = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        group2indices[gid].append(idx)

    group_reps = []
    for idxs in group2indices.values():
        scores = [keyword_score(articles[i]['title'], articles[i]['desc'], keywords) for i in idxs]
        best_idx = idxs[scores.index(max(scores))]
        rep_title = articles[best_idx]['title']
        rep_desc = articles[best_idx]['desc']
        score = max(scores)
        count = len(idxs)
        if count > 1:
            rep_title = f"{rep_title} ({count})"
        summary = rep_desc if len(rep_desc) <= 80 else rep_desc[:80] + "..."
        group_reps.append({
            'title': rep_title,
            'description': summary,
            'score': score,
            'count': count
        })

    # ✅ 중요도 순으로 Top 10 정렬
    top10 = sorted(group_reps, key=lambda x: (x['count'] * 1.6 + x['score']), reverse=True)[:10]

    return top10
