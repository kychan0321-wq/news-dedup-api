from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import functools

app = FastAPI()

# 모델을 전역에서 미리 불러오지 않고, 요청 시 lazy하게 로딩
@functools.lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')


class NewsItem(BaseModel):
    title: str
    description: str

class NewsRequest(BaseModel):
    items: List[NewsItem]

def clean_text(s):
    s = re.sub('<.*?>', ' ', s or '')
    s = re.sub('[^0-9a-zA-Z가-힣 ]+', ' ', s)
    s = re.sub('\s+', ' ', s)
    return s.lower().strip()

keywords = [
    'ai', '인공지능', '로보틱스', '미래산업', '자율주행',
    'sk네트웍스', 'sk networks', 'sk매직', '피닉스랩', 'LLM',
    '생성형 AI', 'Lama', '그록', '일론 머스크', 'sk', 'lg',
    '샘 올트먼', '민팃', '생성형 ai', '삼성'
]

def keyword_score(title, desc, keywords):
    score = 0
    for kw in keywords:
        kw_l = kw.lower()
        score += 2 * title.count(kw_l)
        score += desc.count(kw_l)
    return score

@app.post("/deduplicate")
def deduplicate_news(request: NewsRequest):
    model = get_model()  # 요청이 들어올 때만 모델 로딩

    articles = []
    for item in request.items:
        title = clean_text(item.title)
        desc = clean_text(item.description)
        full = title + ' ' + desc
        articles.append({'title': title, 'desc': desc, 'full': full})

    sentences = [a['full'] for a in articles]
    embeddings = model.encode(sentences, convert_to_numpy=True)

    n = len(embeddings)
    group_ids = [-1] * n
    curr_gid = 0
    thresh = 0.8

    for i in range(n):
        if group_ids[i] == -1:
            group_ids[i] = curr_gid
            for j in range(i + 1, n):
                if group_ids[j] == -1:
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0, 0]
                    if sim >= thresh:
                        group_ids[j] = curr_gid
            curr_gid += 1

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
        summary = rep_desc if len(rep_desc) <= 80 else rep_desc[:80] + "..."
        group_reps.append({
            'title': rep_title,
            'description': summary,
            'score': score,
            'count': len(idxs)
        })

    top_10 = sorted(group_reps, key=lambda x: -x['score'])[:10]

    return top_10
