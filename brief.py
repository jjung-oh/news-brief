#!/usr/bin/env python3
"""
출근길 뉴스 브리핑 생성기 (개인용)

  python3 brief.py collect   # RSS 수집만 (store.json에 누적)
  python3 brief.py build-if-needed   # 오늘 요약이 실패했을 때만 다시 만들기
  python3 brief.py build     # 수집 + 기사 본문 읽기 + Gemini 요약 + docs/index.html 생성

준비: pip install -r requirements.txt  /  환경변수 GEMINI_API_KEY
"""
import difflib
import html as htmllib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import zip_longest
from pathlib import Path
from urllib.parse import unquote
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
# 모델은 자동으로 고릅니다(아래 init_models). 직접 지정하려면 환경변수 GEMINI_MODEL을 설정하세요.
MODELS = []
MIDX = 0
DEAD = set()          # 쓸 수 없는 모델(404, 하루 한도 소진 등)
THINK_LOW = True      # 3.x 모델의 생각량을 낮춰 속도를 높임(지원 안 되면 자동으로 끔)
TIME_LIMIT = 9 * 60   # 요약 전체에 허용하는 최대 시간(초). 넘으면 남은 작업은 건너뜁니다.
DEADLINE = None
MAX_STORIES = 10     # 구간당 카드 수(늘리려면 숫자만 바꾸세요. 많을수록 AI가 실패하기 쉽습니다)
PER_STORY = 3        # 이야기 하나당 읽을 기사 수
BODY_CHARS = 1800    # 기사당 본문 최대 글자 수
MAX_TITLES = 150     # 1단계에 보낼 최대 제목 수
# 날씨 위치(기본: 경기도 광주시). 바꾸려면 위도/경도/이름을 수정하세요.
WEATHER_NAME = os.environ.get("WEATHER_NAME", "경기 광주")
WEATHER_LAT = float(os.environ.get("WEATHER_LAT", "37.4295"))
WEATHER_LON = float(os.environ.get("WEATHER_LON", "127.2550"))
# 기사 시간대(한국 시간, 시 단위)
EVENING_START = 18   # 어제 저녁: 전날 18시 ~ 24시
MORNING_START = 0    # 오늘 아침: 당일 0시 ~ 7시
MORNING_END = 7
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# RSS 주소는 바뀔 수 있습니다. 경고가 뜨면 방송사 RSS 안내 페이지에서 새 주소로 교체하세요.
FEEDS = [
    ("KBS", "https://news.kbs.co.kr/rss/rss.xml"),
    ("MBC", "https://imnews.imbc.com/rss/news/news_00.xml"),
    ("SBS", "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01&plink=RSSREADER"),
    ("SBS", "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=02&plink=RSSREADER"),
    ("SBS", "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=03&plink=RSSREADER"),
    ("SBS", "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=07&plink=RSSREADER"),
    ("JTBC", "https://fs.jtbc.co.kr/RSS/newsflash.xml"),
]


def strip_html(s):
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def load_store():
    if STORE.exists():
        return json.loads(STORE.read_text(encoding="utf-8"))
    return {}


def collect():
    store = load_store()
    now = datetime.now(KST)
    for source, url in FEEDS:
        print(f"수집 중: {source}", flush=True)
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=15)   # 응답이 없으면 15초 뒤 포기
            feed = feedparser.parse(r.content)
            status = r.status_code
        except Exception as e:  # 한 곳이 실패해도 계속 진행
            print(f"[경고] {source} 수집 실패: {e} (주소={url})")
            continue
        if not feed.entries:
            print(f"[경고] {source}: 기사를 가져오지 못했습니다. HTTP={status} 주소={url}")
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
        (f"전날 {EVENING_START}:00–24:00 기사 기준", yday.replace(hour=EVENING_START), today),
        (f"당일 {MORNING_START:02d}:00–{MORNING_END:02d}:00 기사 기준",
         today.replace(hour=MORNING_START), today.replace(hour=MORNING_END)),
    ]


def init_models(client):
    """이 API 키로 실제 사용 가능한 Flash 모델을 조회해 안정 버전·최신 버전 순으로 정렬합니다."""
    global MODELS
    forced = os.environ.get("GEMINI_MODEL")
    if forced:
        MODELS = [forced]
        print("지정된 모델:", forced)
        return
    found = []
    for m in client.models.list():
        name = (m.name or "").replace("models/", "")
        acts = getattr(m, "supported_actions", None)
        if acts and "generateContent" not in acts:
            continue
        mt = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash(-lite)?(-preview[\w-]*)?", name)
        if mt:   # 일반 Flash 먼저(안정판 > 최신), lite는 맨 뒤 예비용
            found.append((0 if mt.group(2) else 1, 0 if mt.group(3) else 1, float(mt.group(1)), name))
    found.sort(reverse=True)
    MODELS = [n for *_, n in found]
    print("사용 가능한 Flash 모델(우선순위):", MODELS)
    if not MODELS:
        raise RuntimeError("사용 가능한 Gemini Flash 모델을 찾지 못했습니다.")


def next_model():
    """살아 있는 다음 모델로 이동(끝에 닿으면 처음으로). 없으면 False."""
    global MIDX
    for _ in range(len(MODELS)):
        MIDX = (MIDX + 1) % len(MODELS)
        if MODELS[MIDX] not in DEAD:
            return True
    return False


def ask_json(client, prompt, max_tokens=None):
    """Gemini에 JSON으로 답하게 요청합니다.
    - 모델이 없거나 하루 한도를 다 쓴 경우: 그 모델을 제외하고 다른 모델로
    - 서버 혼잡(429/5xx): 모델을 돌려가며 기다렸다 다시 시도(최대 6번, 전체 제한 시간 안에서)"""
    global THINK_LOW
    if not MODELS:
        init_models(client)
    last, attempts = None, 0
    while attempts < 6:
        if DEADLINE and time.time() > DEADLINE:
            raise RuntimeError("제한 시간(9분)을 넘겨 요약을 건너뜁니다.")
        while MODELS[MIDX] in DEAD:
            if not next_model():
                raise RuntimeError("사용 가능한 Gemini 모델이 없습니다.")
        model = MODELS[MIDX]
        try:
            cfg = {"response_mime_type": "application/json", "temperature": 0.3}
            if max_tokens:
                cfg["max_output_tokens"] = max_tokens
            if model.startswith("gemini-2.5"):
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            elif THINK_LOW and model.startswith("gemini-3") and "lite" not in model:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
            resp = client.models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
            )
            text = (resp.text or "").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            return json.loads(text)
        except Exception as e:
            last = e
            msg = str(e)
            print(f"[재시도] Gemini 호출 실패({model}): {msg[:160]}")
            if "API key" in msg or "PERMISSION_DENIED" in msg:
                raise
            if THINK_LOW and "thinking" in msg.lower():
                THINK_LOW = False
                print("[설정] 생각량 낮춤 옵션을 끄고 다시 시도합니다.")
                continue
            if "NOT_FOUND" in msg or "limit: 0" in msg or "PerDay" in msg or "per day" in msg:
                DEAD.add(model)
                print(f"[모델 제외] {model}")
                continue
            attempts += 1
            if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "500", "503", "504",
                                      "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL")):
                next_model()
                print(f"[모델 변경] {MODELS[MIDX]} (서버 혼잡 {attempts}/6)")
            time.sleep(min(15 * attempts, 45))
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
    res = ask_json(client, PICK_PROMPT.format(n=MAX_STORIES, k=PER_STORY, lines=lines), 3000)
    picked = res.get("stories", []) if isinstance(res, dict) else res   # 목록으로 와도 처리
    stories = []
    for s in picked[:MAX_STORIES]:
        if not isinstance(s, dict):
            continue
        ids = [i for i in s.get("ids", []) if isinstance(i, int) and 1 <= i <= len(arts)]
        if ids:
            stories.append({"c": s.get("c", ""), "arts": [arts[i - 1] for i in ids[:PER_STORY]]})
    if not stories:
        raise RuntimeError("AI가 고른 기사가 없습니다.")

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

    res = ask_json(
        client, SUM_PROMPT.format(m=len(stories), blocks="\n\n".join(blocks)), 16000
    )
    items = res.get("items", []) if isinstance(res, dict) else res
    items = [it for it in items if isinstance(it, dict)]

    # 원문 링크는 AI가 아니라 스크립트가 직접 붙입니다(링크 변조 방지).
    if len(items) == len(stories):
        for it, st in zip(items, stories):
            it["links"] = [{"s": a["source"], "u": a["link"]} for a in st["arts"]]
    return items


KMA_KEY = os.environ.get("KMA_API_KEY", "").strip()   # 선택: 기상청 공식 예보용 키
KMA_URLS = [  # (주소, 키 파라미터 이름) - API허브 키와 공공데이터포털 키를 모두 시도합니다
    ("https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst", "authKey"),
    ("https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst", "serviceKey"),
]

WMO = {
    0: "맑음", 1: "대체로 맑음", 2: "구름 조금", 3: "흐림", 45: "안개", 48: "안개",
    51: "약한 이슬비", 53: "이슬비", 55: "강한 이슬비", 56: "어는 이슬비", 57: "어는 이슬비",
    61: "약한 비", 63: "비", 65: "강한 비", 66: "어는 비", 67: "어는 비",
    71: "약한 눈", 73: "눈", 75: "강한 눈", 77: "싸락눈",
    80: "약한 소나기", 81: "소나기", 82: "강한 소나기", 85: "눈 소나기", 86: "강한 눈 소나기",
    95: "뇌우", 96: "우박 동반 뇌우", 99: "우박 동반 뇌우",
}


def grade(v, cuts):
    if v is None:
        return None
    return "좋음" if v <= cuts[0] else "보통" if v <= cuts[1] else "나쁨" if v <= cuts[2] else "매우 나쁨"


def latlon_to_grid(lat, lon):
    """위도/경도 -> 기상청 동네예보 격자(nx, ny) (기상청 공개 변환식)."""
    RE, GRID, SLAT1, SLAT2, OLON, OLAT, XO, YO = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    d = math.pi / 180.0
    re_ = RE / GRID
    s1, s2, ol, oa = SLAT1 * d, SLAT2 * d, OLON * d, OLAT * d
    sn = math.log(math.cos(s1) / math.cos(s2)) / math.log(
        math.tan(math.pi * 0.25 + s2 * 0.5) / math.tan(math.pi * 0.25 + s1 * 0.5))
    sf = math.pow(math.tan(math.pi * 0.25 + s1 * 0.5), sn) * math.cos(s1) / sn
    ro = re_ * sf / math.pow(math.tan(math.pi * 0.25 + oa * 0.5), sn)
    ra = re_ * sf / math.pow(math.tan(math.pi * 0.25 + lat * d * 0.5), sn)
    th = lon * d - ol
    th = th - 2 * math.pi if th > math.pi else th + 2 * math.pi if th < -math.pi else th
    th *= sn
    return math.floor(ra * math.sin(th) + XO + 0.5), math.floor(ro - ra * math.cos(th) + YO + 0.5)


def air_quality_line():
    try:
        aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=15, params={
            "latitude": WEATHER_LAT, "longitude": WEATHER_LON, "timezone": "Asia/Seoul",
            "forecast_days": 1, "hourly": "pm10,pm2_5"}).json()["hourly"]
        pm10 = max([v for v in aq["pm10"][7:21] if v is not None], default=None)
        pm25 = max([v for v in aq["pm2_5"][7:21] if v is not None], default=None)
        g10, g25 = grade(pm10, (30, 80, 150)), grade(pm25, (15, 35, 75))
        if g10 and g25:
            return f"미세먼지 {g10}, 초미세먼지 {g25} (Open-Meteo 모델 예측치, 낮 시간 최대 기준)"
    except Exception as e:
        print(f"[경고] 미세먼지 정보 실패: {e}")
    return None


def make_card(desc, tmin, tmax, t8, t18, pop, rain_pt, wind, src):
    tip = ("우산을 꼭 챙기세요." if pop >= 60 else
           "혹시 모르니 우산을 챙기세요." if pop >= 30 else "우산은 필요 없겠습니다.")
    sentence = (f"오늘 {WEATHER_NAME}은 {desc}, 최저 {tmin}도 최고 {tmax}도입니다. "
                f"출근 시간대인 오전 8시에는 {t8}도입니다. 강수 확률은 최대 {pop}퍼센트로, {tip}")
    pts = [
        f"기온: 최저 {tmin}도, 최고 {tmax}도 (오전 8시 {t8}도, 오후 6시 {t18}도)",
        f"강수: 확률 최대 {pop}퍼센트, {rain_pt}",
        f"바람: 최대 풍속 약 {wind:.0f}미터" + (" — 바람이 강하니 주의하세요" if wind >= 9 else ""),
    ]
    if tmax - tmin >= 10:
        pts.append(f"일교차가 {tmax - tmin}도로 큽니다. 겉옷을 챙기세요")
    aq = air_quality_line()
    if aq:
        pts.append(aq)
    return {"c": "날씨", "h": f"오늘 {WEATHER_NAME} 날씨 {desc}, 최저 {tmin}도 최고 {tmax}도",
            "s": sentence, "pts": pts, "kw": [WEATHER_NAME, desc, "날씨"], "src": src}


def weather_kma():
    """기상청 단기예보(공식). 키가 필요합니다."""
    now = datetime.now(KST)
    nx, ny = latlon_to_grid(WEATHER_LAT, WEATHER_LON)
    if now.hour < 2 or (now.hour == 2 and now.minute < 20):
        bd, bt = (now - timedelta(days=1)).strftime("%Y%m%d"), "2300"
    else:
        bd, bt = now.strftime("%Y%m%d"), "0200"
    items, errs = None, []
    for url, keyname in KMA_URLS:
        try:
            r = requests.get(url, timeout=20, params={
                keyname: unquote(KMA_KEY), "pageNo": 1, "numOfRows": 1500, "dataType": "JSON",
                "base_date": bd, "base_time": bt, "nx": nx, "ny": ny})
            try:
                j = r.json()
            except Exception:
                raise RuntimeError(f"JSON 아님: {r.text[:150]}")
            if j["response"]["header"]["resultCode"] != "00":
                raise RuntimeError(j["response"]["header"].get("resultMsg", "오류"))
            items = j["response"]["body"]["items"]["item"]
            print(f"기상청 예보 사용: {url.split('/')[2]} (격자 {nx},{ny}, 기준 {bd} {bt})")
            break
        except Exception as e:
            errs.append(f"{url.split('/')[2]}: {e}")
    if items is None:
        raise RuntimeError(" / ".join(errs))

    today = now.strftime("%Y%m%d")
    by = {}
    for it in items:
        if it["fcstDate"] == today:
            by.setdefault(it["category"], {})[it["fcstTime"]] = it["fcstValue"]

    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    tmp = {t: f(v) for t, v in by.get("TMP", {}).items() if f(v) is not None}
    tmin = f(next(iter(by.get("TMN", {}).values()), None))
    tmax = f(next(iter(by.get("TMX", {}).values()), None))
    tmin = round(tmin if tmin is not None else min(tmp.values()))
    tmax = round(tmax if tmax is not None else max(tmp.values()))
    t8, t18 = round(tmp["0800"]), round(tmp["1800"])
    pop = int(max([f(v) or 0 for v in by.get("POP", {}).values()], default=0))
    wind = max([f(v) or 0 for v in by.get("WSD", {}).values()], default=0)

    pcp = [v for v in by.get("PCP", {}).values() if v and v != "강수없음"]
    num = lambda v: float(re.search(r"[\d.]+", v).group()) if re.search(r"[\d.]+", v) else 0.0
    rain_pt = ("시간당 최대 강수량 " + max(pcp, key=num).replace("mm", "밀리미터")) if pcp else "강수 없음"

    pty = [v for v in by.get("PTY", {}).values() if v != "0"]
    if pty:
        desc = {"1": "비", "2": "비 또는 눈", "3": "눈", "4": "소나기", "5": "빗방울",
                "6": "빗방울과 눈날림", "7": "눈날림"}.get(Counter(pty).most_common(1)[0][0], "강수")
    else:
        sky = [v for t, v in by.get("SKY", {}).items() if "0900" <= t <= "1800"] or list(by.get("SKY", {}).values())
        desc = {"1": "맑음", "3": "구름 많음", "4": "흐림"}.get(Counter(sky).most_common(1)[0][0], "변화 있음")
    return make_card(desc, tmin, tmax, t8, t18, pop, rain_pt, wind, "기상청 단기예보")


def weather_openmeteo():
    common = {"latitude": WEATHER_LAT, "longitude": WEATHER_LON, "timezone": "Asia/Seoul", "forecast_days": 1}
    w = requests.get("https://api.open-meteo.com/v1/forecast", timeout=15, params={
        **common,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                 "precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
        "hourly": "temperature_2m,precipitation_probability",
    }).json()
    if "daily" not in w:
        raise RuntimeError(w.get("reason", "날씨 서버 응답 이상"))
    d, h = w["daily"], w["hourly"]
    return make_card(
        WMO.get(d["weather_code"][0], "변화 있음"),
        round(d["temperature_2m_min"][0]), round(d["temperature_2m_max"][0]),
        round(h["temperature_2m"][8]), round(h["temperature_2m"][18]),
        d["precipitation_probability_max"][0] or 0,
        f"예상 강수량 {(d['precipitation_sum'][0] or 0):.1f}밀리미터",
        (d["wind_speed_10m_max"][0] or 0) / 3.6,
        "Open-Meteo 예측 모델(기상청 공식 예보와 다를 수 있음)",
    )


def fetch_weather():
    """KMA_API_KEY가 있으면 기상청 공식 예보, 없거나 실패하면 Open-Meteo로 대체합니다."""
    if KMA_KEY:
        try:
            return weather_kma()
        except Exception as e:
            print(f"[경고] 기상청 예보 실패, Open-Meteo로 대체합니다: {e}")
    else:
        print("KMA_API_KEY가 없어 Open-Meteo 날씨를 사용합니다.")
    for i in range(3):
        try:
            return weather_openmeteo()
        except Exception as e:
            print(f"[경고] 날씨 정보 실패({i + 1}/3): {e}")
            time.sleep(3 * (i + 1))
    return None


def old_weather(today_s):
    """오늘 이미 만들어 둔 날씨 카드가 있으면 재사용합니다(날씨 서버가 잠깐 막혔을 때 대비)."""
    try:
        old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for it in old[1]["items"]:
            if it.get("c") == "날씨" and it.get("date") == today_s and it.get("pts"):
                return it
    except Exception:
        pass
    return None


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or ""))
    return [x.strip() for x in parts if len(x.strip()) >= 15]


def pick_diverse(arts, n):
    """방송사를 번갈아 가며 최신 기사부터 고르고, 제목이 비슷한 중복은 건너뜁니다."""
    by = {}
    for a in sorted(arts, key=lambda a: a["ts"], reverse=True):
        by.setdefault(a["source"], []).append(a)
    out, seen = [], []
    for group in zip_longest(*by.values()):
        for a in group:
            if a is None:
                continue
            t = htmllib.unescape(a["title"])
            if any(difflib.SequenceMatcher(None, t, x).ratio() > 0.6 for x in seen):
                continue
            seen.append(t)
            out.append(a)
            if len(out) >= n:
                return out
    return out


def headline_items(arts, n=None):
    """AI가 안 될 때의 대체: 기사 본문 앞부분을 그대로 요약처럼 보여줍니다(AI 없음)."""
    chosen = pick_diverse(arts, n or MAX_STORIES)
    with ThreadPoolExecutor(max_workers=8) as ex:
        bodies = list(ex.map(lambda a: fetch_body(a["link"]), chosen))
    out = []
    for a, body in zip(chosen, bodies):
        sents = sentences(body)
        lead = " ".join(sents[:2])[:220] if sents else htmllib.unescape(a["summary"])
        out.append({
            "c": "뉴스", "h": htmllib.unescape(a["title"]),
            "s": lead or "원문 링크에서 자세히 볼 수 있어요.",
            "pts": [x[:120] for x in sents[2:5]],
            "kw": [a["source"]], "src": a["source"],
            "links": [{"s": a["source"], "u": a["link"]}],
        })
    return out


def needs_build():
    """오늘 요약이 이미 모두 성공했으면 False(재시도 실행을 건너뛰기 위함)."""
    try:
        old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        today_s = datetime.now(KST).strftime("%Y-%m-%d")
        return not (old[0].get("built") == today_s and all(w.get("ok") for w in old))
    except Exception:
        return True


def build():
    store = collect()
    now = datetime.now(KST)
    built = now.strftime("%Y-%m-%d")
    global DEADLINE
    DEADLINE = time.time() + TIME_LIMIT
    client = genai.Client(http_options=types.HttpOptions(timeout=60_000))  # 호출 1회 최대 60초
    data = []
    ai_down = False
    for label, start, end in windows(now):
        arts = [
            {**v, "link": k}
            for k, v in store.items()
            if start <= datetime.fromisoformat(v["ts"]) < end
        ]
        print(f"{label}: 기사 {len(arts)}건 처리 중...")
        ok = True
        try:
            if ai_down and arts:
                raise RuntimeError("Gemini 서버 혼잡이 이어져 AI 요약을 건너뜁니다(사용 가능한 모델 없음).")
            items = process_window(client, arts)
        except Exception as e:
            print(f"[오류] {label} 요약 실패: {str(e)[:300]}")
            ok = False
            if any(k in str(e) for k in ("503", "504", "UNAVAILABLE", "DEADLINE", "429", "제한 시간", "사용 가능한", "혼잡")):
                ai_down = True
            items = [{
                "c": "알림", "h": "AI 요약이 안 돼서 기사 앞부분을 보여드려요",
                "s": "Gemini 서버가 혼잡하거나 사용량이 부족해요. 아래는 AI 없이 기사 본문의 앞부분을 그대로 가져온 거예요. 나중에 자동으로 다시 시도해요.",
                "pts": ["Actions 로그의 [오류] 줄에서 자세한 원인을 볼 수 있어요"],
                "kw": ["알림"], "src": "시스템",
            }] + headline_items(arts)
        data.append({"label": label, "items": items, "ok": ok})

    data[0]["built"] = built
    today_s = built
    weather = fetch_weather()
    if weather:
        weather["date"] = today_s
    else:
        weather = old_weather(today_s) or {
            "c": "날씨", "h": "날씨 정보를 가져오지 못했어요",
            "s": "날씨 서버가 잠시 응답하지 않았어요. 다음 실행 때 다시 가져올게요.",
            "pts": [], "kw": ["날씨"], "src": "시스템",
        }
    data[1]["items"].insert(0, weather)

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
    sys.stdout.reconfigure(line_buffering=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "collect":
        collect()
    elif mode == "build":
        build()
    elif mode == "build-if-needed":
        if needs_build():
            build()
        else:
            print("오늘 요약이 이미 완성되어 있어 건너뜁니다.")
    else:
        print(__doc__)
