#!/usr/bin/env python3
"""
출근길 뉴스 브리핑 생성기 (개인용)

  python3 brief.py collect   # RSS 수집만 (store.json에 누적)
  python3 brief.py build     # 수집 + 기사 본문 읽기 + Gemini 요약 + docs/index.html 생성

준비: pip install -r requirements.txt  /  환경변수 GEMINI_API_KEY
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
import requests
import trafilatura
from google import genai
from google.genai import types

KST = ZoneInfo("Asia/Seoul")
BASE = Path(__file__).parent
STORE = BASE / "store.json"
TEMPLATE = BASE / "news-brief.html"
OUT_JSON = BASE / "news.json"
OUT_HTML = BASE / "docs" / "index.html"
# Gemini 무료 구간에서 쓸 수 있는 Flash 계열 모델. 안 되면 AI Studio의 모델 목록에서 이름을 확인해 GEMINI_MODEL로 바꾸세요.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
MAX_STORIES = 8      # 구간당 카드 수
PER_STORY = 3        # 이야기 하나당 읽을 기사 수
BODY_CHARS = 1800    # 기사당 본문 최대 글자 수
MAX_TITLES = 150     # 1단계에 보낼 최대 제목 수
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# RSS 주소는 바뀔 수 있습니다. 경고가 뜨면 방송사 RSS 안내 페이지에서 새 주소로 교체하세요.
FEEDS = {
    "KBS": "https://news.kbs.co.kr/rss/rss.xml",
    "MBC": "https://imnews.imbc.com/rss/news/news_00.xml",
    "SBS": "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01&plink=RSSREADER",
    "JTBC": "https://fs.jtbc.co.kr/RSS/newsflash.xml",
}


def strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def load_store():
    if STORE.exists():
        return json.loads(STORE.read_text(encoding="utf-8"))
    return {}


def collect():
    store = load_store()
    now = datetime.now(KST)
    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url, agent=UA)
        except Exception as e:
            print(f"[경고] {source} 수집 실패: {e}")
            continue
        if not feed.entries:
            print(f"[경고] {source}: 기사를 가져오지 못했습니다. RSS 주소를 확인하세요.")
        for e in feed.entries:
            link = e.get("link")
            if not link or link in store:
                continue
            t = e.get("published_parsed") or e.get("updated_parsed")
            ts = datetime(*t[:6], tzinfo=timezone.utc).astimezone(KST) if t else now
            store[link] = {
                "source": source,
                "title": strip_html(e.get("title", "")),
                "summary": strip_html(e.get("summary", ""))[:300],
                "ts": ts.isoformat(),
            }
    cutoff = now - timedelta(days=3)
    store = {k: v for k, v in store.items() if datetime.fromisoformat(v["ts"]) >= cutoff}
    STORE.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"수집 완료: 누적 기사 {len(store)}건")
    return store


def windows(now):
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yday = today - timedelta(days=1)
    return [
        ("전날 18:00–24:00 기사 기준", yday.replace(hour=18), today),
        ("당일 05:00–09:00 기사 기준", today.replace(hour=5), today.replace(hour=9)),
    ]


def ask_json(client, prompt, max_tokens=None):
    """Gemini에 JSON으로 답하게 요청. 무료 구간의 429(요청 한도) 오류는 기다렸다가 재시도합니다."""
    last = None
    for wait in (0, 30, 60, 90):
        time.sleep(wait)
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.3
                ),
            )
            text = (resp.text or "").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            return json.loads(text)
        except Exception as e:
            last = e
            print(f"[재시도] Gemini 호출 실패: {e}")
    raise last


PICK_PROMPT = """아래는 방송사 기사 제목 목록입니다. 출근길에 꼭 알아야 할 이야기를 중요도 순으로 최대 {n}개 고르세요.
- 같은 사건을 다룬 기사는 하나의 이야기로 묶고, ids에 해당 번호를 넣으세요(최대 {k}개).
- 분야(c)는 정치, 경제, 사회, 국제, 문화·스포츠, 날씨 중 하나입니다.
- 설명 없이 JSON만 출력: {{"stories":[{{"c":"","ids":[1,2]}}]}}

기사 목록:
{lines}
"""

SUM_PROMPT = """아래 {m}개의 이야기를 출근길에 듣기 좋은 뉴스 브리핑으로 정리하세요.
- 반드시 제공된 본문/요약에 있는 내용만 사용하고, 없는 사실이나 숫자를 만들지 마세요.
- 입력 순서 그대로 정확히 {m}개의 항목을 출력하세요.
- c: 분야 / h: 40자 이내 헤드라인 / s: 소리 내어 읽기 좋은 2~3문장 요약
- pts: 핵심 포인트 3~4개(각 한 문장, 배경·원인·전망 포함) / kw: 키워드 2~4개 / src: 방송사 이름들
- 설명 없이 JSON만 출력: {{"items":[{{"c":"","h":"","s":"","pts":[""],"kw":[""],"src":""}}]}}

{blocks}
"""


def fetch_body(url):
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=10)
        r.raise_for_status()
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
        return text[:BODY_CHARS]
    except Exception:
        return ""


def process_window(client, arts):
    if not arts:
        return []
    arts = sorted(arts, key=lambda a: a["ts"])[-MAX_TITLES:]

    # 1단계: 제목만 보고 중요한 이야기 고르기
    lines = "\n".join(f"{i}. [{a['source']}] {a['title']}" for i, a in enumerate(arts, 1))
    picked = ask_json(
        client, PICK_PROMPT.format(n=MAX_STORIES, k=PER_STORY, lines=lines), 2000
    )["stories"]
    stories = []
    for s in picked[:MAX_STORIES]:
        ids = [i for i in s.get("ids", []) if isinstance(i, int) and 1 <= i <= len(arts)]
        if ids:
            stories.append({"c": s.get("c", ""), "arts": [arts[i - 1] for i in ids[:PER_STORY]]})

    # 2단계: 고른 기사 본문만 가져오기
    urls = [a["link"] for st in stories for a in st["arts"]]
    with ThreadPoolExecutor(max_workers=8) as ex:
        bodies = dict(zip(urls, ex.map(fetch_body, urls)))

    blocks = []
    for k, st in enumerate(stories, 1):
        parts = [f"## 이야기 {k} (분야: {st['c']})"]
        for a in st["arts"]:
            body = bodies.get(a["link"]) or f"(본문 수집 실패, RSS 요약) {a['summary']}"
            parts.append(f"[{a['source']}] {a['title']}\n본문: {body}")
        blocks.append("\n".join(parts))

    items = ask_json(
        client, SUM_PROMPT.format(m=len(stories), blocks="\n\n".join(blocks)), 6000
    )["items"]

    # 원문 링크는 AI가 아니라 스크립트가 직접 붙입니다(링크 변조 방지).
    if len(items) == len(stories):
        for it, st in zip(items, stories):
            it["links"] = [{"s": a["source"], "u": a["link"]} for a in st["arts"]]
    return items


def build():
    store = collect()
    now = datetime.now(KST)
    client = genai.Client()  # GEMINI_API_KEY 환경변수 사용
    data = []
    for label, start, end in windows(now):
        arts = [
            {**v, "link": k}
            for k, v in store.items()
            if start <= datetime.fromisoformat(v["ts"]) < end
        ]
        print(f"{label}: 기사 {len(arts)}건 처리 중...")
        try:
            items = process_window(client, arts)
        except Exception as e:
            print(f"[오류] {label} 요약 실패: {e}")
            items = []
        data.append({"label": label, "items": items})

    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    if not TEMPLATE.exists():
        print("news-brief.html 템플릿이 없어 news.json만 만들었습니다.")
        return
    html = TEMPLATE.read_text(encoding="utf-8")
    payload = "var DATA=" + json.dumps(data, ensure_ascii=False).replace("</", "<\\/") + ";"
    html = re.sub(r"^var DATA=.*;$", lambda m: payload, html, count=1, flags=re.M)
    html = html.replace(
        "⚠️ 기사는 <b>샘플 문구</b>입니다. 카드를 눌러 자세히 보고, 🔊로 들어보세요.",
        "생성 시각 " + now.strftime("%m/%d %H:%M") + " · 카드를 눌러 자세히 보고, 🔊로 들어보세요.",
    )
    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"완료: {OUT_HTML}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "collect":
        collect()
    elif mode == "build":
        build()
    else:
        print(__doc__)
