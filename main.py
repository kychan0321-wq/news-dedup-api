from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from collections import defaultdict
import re

app = FastAPI()

class NewsItem(BaseModel):
    title: str
    description: str

class NewsRequest(BaseModel):
    items: List[NewsItem]

# 텍스트 전처리 함수
def clean_text(s):
    s = re.sub('<.*?>', ' ', s or '')
    s = re.sub('[^0-9a-zA-Z가-힣 ]+', ' ', s)
    s = re.sub('\s+', ' ', s)
    return s.lower().strip()

# 공통 단어 수 기반 유사도 계산
def is_similar(text1, text2, thresh=0.5):
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 or not words2:
        return False
    overlap = words1 & words2
    score = len(overlap) / max(len(words1), len(words2))
    return score >= thresh

keywords = [
    'ai', '인공지능', '로보틱스', '미래산업', '자율주행',
    'sk네트웍스', 'sk networks', 'sk매직', '피닉스랩', 'LLM',
    '생성형 AI', 'Lama', '그록', '일론 머스크', 'sk', 'lg',
    '샘 올트먼', '민팃', '생성형 ai', '삼성'
]

# 키워드 포함 개수로 간단한 score 측정
def keyword_score(text, keywords):
    return sum(text.count(kw.lower()) for kw in keywords)

@app.post("/deduplicate")
def deduplicate_news(request: NewsRequest):
    raw_articles = []
    for item in request.items:
        title = clean_text(item.title)
        desc = clean_text(item.description)
        full = title + ' ' + desc
        raw_articles.append({'title': title, 'desc': desc, 'full': full})

    # 유사도 기반 그룹핑
    n = len(raw_articles)
    group_ids = [-1] * n
    curr_gid = 0

    for i in range(n):
        if group_ids[i] == -1:
            group_ids[i] = curr_gid
            for j in range(i + 1, n):
                if group_ids[j] == -1:
                    if is_similar(raw_articles[i]['full'], raw_articles[j]['full']):
                        group_ids[j] = curr_gid
            curr_gid += 1

    # 그룹별 대표 기사 선정 (중복 수 + 키워드 포함 수 기준)
    group2indices = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        group2indices[gid].append(idx)

    group_reps = []
    for idxs in group2indices.values():
        best_idx = max(idxs, key=lambda i: keyword_score(raw_articles[i]['full'], keywords))
        rep = raw_articles[best_idx]
        group_reps.append({
            'title': rep['title'],
            'description': rep['desc'][:80] + ('...' if len(rep['desc']) > 80 else ''),
            'count': len(idxs),
            'score': keyword_score(rep['full'], keywords)
        })

    # 중요도: 중복 수 * 2 + 키워드 score로 정렬
    top_10 = sorted(group_reps, key=lambda x: (x['count'] * 2 + x['score']), reverse=True)[:10]

    return top_10
