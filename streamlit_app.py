# Streamlit app:
#   - Tab 1: Scrape https://www.mlforseo.com/experts/ grid (requests + BeautifulSoup)
#   - Tab 2: Verify authorship of submitted content URLs against a provided author name
#
# Notes:
# - Author check gathers evidence from JSON-LD, meta tags, visible bylines, domain adapters
# - LinkedIn Pulse often blocks bots; we add slug heuristics + JSON-LD regex fallback
# - Matching uses normalized tokens + fuzzy ratio (difflib) to classify MATCH/POSSIBLE/NO_MATCH

import re
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import streamlit as st

# Optional deps
try:
    from rapidfuzz import fuzz
    HAVE_RF = True
except Exception:
    HAVE_RF = False

try:
    from requests_html import HTMLSession
    HAVE_RHTML = True
except Exception:
    HAVE_RHTML = False


APP_NAME = "Authorship Verifier"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]
DEFAULT_HEADERS = {
    "User-Agent": random.choice(USER_AGENTS),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
}

BYLINE_CLASSES = re.compile(r"(byline|author|contributor|writer|posted-by)", re.I)
BYLINE_REGEX = re.compile(r"\bby\s+([A-Z][\w'’\-. ]{1,80})\b", re.I)


# ----------------- Small model -----------------

@dataclass
class Signal:
    source: str
    value: str
    weight: float
    note: str


# ----------------- Utils -----------------

def norm(s: str) -> str:
    return (s or "").strip()

def norm_name(s: str) -> str:
    s = norm(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s

def sim(a: str, b: str) -> float:
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    if HAVE_RF:
        return fuzz.token_set_ratio(a, b) / 100.0
    sa, sb = set(re.findall(r"[a-z0-9]+", a)), set(re.findall(r"[a-z0-9]+", b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

@st.cache_data(show_spinner=False)
def fetch(url: str, timeout: int = 20, render_js: bool = False) -> Tuple[int, str, str]:
    """Return (status, text, final_url). JS render only if requests-html is available."""
    try:
        if render_js and HAVE_RHTML:
            session = HTMLSession()
            r = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            try:
                r.html.render(timeout=timeout)
                text = r.html.html or r.text
            except Exception:
                text = r.text
            final_url = str(r.url)
            status = r.status_code
            session.close()
            return status, text, final_url
        else:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            return r.status_code, r.text, r.url
    except Exception as e:
        return 0, f"__ERROR__ {e}", url

def parse_html(html_text: str) -> BeautifulSoup:
    return BeautifulSoup(html_text, "lxml")


# ----------------- JSON-LD safe helpers (NEW) -----------------

def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _extract_urlish(x):
    """Return a URL-like string from JSON-LD that may be str/dict/list."""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for k in ("@id", "url", "id", "mainEntityOfPage"):
            s = _extract_urlish(x.get(k))
            if s:
                return s
    if isinstance(x, list):
        for v in x:
            s = _extract_urlish(v)
            if s:
                return s
    return ""

def _extract_textish(x):
    """Return text-like string from JSON-LD that may be str/dict/list."""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for k in ("headline", "name", "title"):
            s = _extract_textish(x.get(k))
            if s:
                return s
    if isinstance(x, list):
        for v in x:
            s = _extract_textish(v)
            if s:
                return s
    return ""

def _collect_types(obj):
    """Flatten @type to a lowercase set."""
    out = set()
    if isinstance(obj, dict):
        t = obj.get("@type")
        if isinstance(t, str):
            out.add(t.lower())
        elif isinstance(t, list):
            out.update([str(x).lower() for x in t])
    return out

def _flatten_jsonld_entities(blob):
    """Yield dicts from a JSON-LD script (handles @graph/arrays)."""
    for item in _as_list(blob):
        if isinstance(item, dict):
            if "@graph" in item and isinstance(item["@graph"], list):
                for g in item["@graph"]:
                    if isinstance(g, dict):
                        yield g
            else:
                yield item

def _extract_authors_from_jsonld(obj):
    """Return list of author-like names from JSON-LD (author/creator variants)."""
    names = []
    def collect(val):
        if isinstance(val, str):
            s = val.strip()
            if s:
                names.append(s)
        elif isinstance(val, dict):
            n = _extract_textish(val.get("name"))
            if n:
                names.append(n)
        elif isinstance(val, list):
            for v in val:
                collect(v)
    for key in ("author", "creator"):
        if key in obj:
            collect(obj[key])
    # de-dup while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def _url_same_page(a, b):
    """Loose same-page check: compare netloc+path (ignore query/fragment)."""
    from urllib.parse import urlparse
    if not a or not b:
        return False
    pa, pb = urlparse(a), urlparse(b)
    return (pa.netloc.lower(), pa.path.rstrip("/")) == (pb.netloc.lower(), pb.path.rstrip("/"))


# ----------------- Signal extraction -----------------

def extract_author_signals(html_text: str, page_url: str) -> List[Signal]:
    if html_text.startswith("__ERROR__"):
        return [Signal("error", "", 0.0, html_text)]

    soup = parse_html(html_text)
    signals: List[Signal] = []

    # 1) Meta tags (strong)
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"):
        signals.append(Signal("meta[name=author]", meta_author["content"], 1.0, "Direct author meta"))

    for prop in ["article:author", "og:article:author", "og:author"]:
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            signals.append(Signal(f"meta[property={prop}]", tag["content"], 0.95, "OG/article author"))

    tw_creator = soup.find("meta", attrs={"name": "twitter:creator"})
    if tw_creator and tw_creator.get("content"):
        handle = tw_creator["content"].lstrip("@")
        if handle:
            signals.append(Signal("meta[name=twitter:creator]", handle, 0.6, "Twitter handle"))

    # 2) JSON-LD (robust)
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.text or ""
        if not raw.strip():
            continue
        try:
            blob = json.loads(raw)
        except Exception:
            # try split-objects fallback
            parts = re.split(r"(?<=\})\s*(?=\{)", raw.strip())
            blob = []
            for p in parts:
                p = p.strip().rstrip(",")
                try:
                    blob.append(json.loads(p))
                except Exception:
                    pass
        for obj in _flatten_jsonld_entities(blob):
            types = _collect_types(obj)
            is_articleish = bool(types & {"article", "blogposting", "newsarticle", "videoobject"})
            obj_url = _extract_urlish(obj.get("url")) or _extract_urlish(obj.get("mainEntityOfPage"))
            # headline = _extract_textish(obj.get("headline")) or _extract_textish(obj.get("name"))  # unused but kept for future heuristics
            same_page = _url_same_page(obj_url, page_url) if obj_url else False

            author_names = _extract_authors_from_jsonld(obj)
            if not author_names:
                continue
            if is_articleish and (same_page or not obj_url):
                for n in author_names:
                    signals.append(Signal("jsonld", n, 1.1, "Schema.org author"))
            else:
                for n in author_names:
                    signals.append(Signal("jsonld-weak", n, 0.8, "Schema.org author (weak tie)"))

    # 3) Byline heuristics
    for el in soup.find_all(True, attrs={"class": BYLINE_CLASSES}):
        txt = el.get_text(" ", strip=True)
        if txt:
            signals.append(Signal("byline-class", txt, 0.75, "Byline container"))
            m = BYLINE_REGEX.search(txt)
            if m:
                signals.append(Signal("byline-regex", m.group(1), 0.9, "Byline ‘by …’"))

    # 4) rel=author
    for a in soup.find_all("a", attrs={"rel": re.compile(r"\bauthor\b", re.I)}):
        val = a.get_text(" ", strip=True)
        if val:
            signals.append(Signal("rel=author", val, 0.85, "Rel author link"))

    # 5) Title hint (weak)
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        if title:
            signals.append(Signal("title", title, 0.25, "Weak heuristic from <title>"))

    # Deduplicate near-identical values per source+value
    dedup = {}
    for s in signals:
        key = (s.source, norm_name(s.value))
        if key not in dedup or s.weight > dedup[key].weight:
            dedup[key] = s
    return list(dedup.values())


def judge_author_match(author_name: str, signals: List[Signal]) -> Tuple[float, Optional[Signal], List[Signal]]:
    author_name = norm(author_name)
    best_s = None
    best_score = 0.0

    for s in signals:
        if not s.value:
            continue
        base = s.weight
        score = sim(author_name, s.value) * base
        # If class/title blob, try extracting “by …”
        if score < 0.4 and s.source in ("title", "byline-class") and " by " in norm_name(s.value):
            m = BYLINE_REGEX.search(s.value)
            if m:
                score = max(score, sim(author_name, m.group(1)) * (base + 0.05))
        if score > best_score:
            best_score = score
            best_s = s

    return min(1.0, best_score), best_s, signals


def classify_conf(score: float) -> Tuple[str, str]:
    if score >= 0.85:
        return "✅ Match", f"{score:.2f}"
    if score >= 0.65:
        return "🟨 Likely", f"{score:.2f}"
    if score >= 0.45:
        return "🟧 Unclear", f"{score:.2f}"
    return "❌ No Match", f"{score:.2f}"


def verify_authorship(author_name: str, url: str, use_js: bool = False) -> Dict:
    status, html_text, final_url = fetch(url, render_js=use_js)
    if status == 0 or html_text.startswith("__ERROR__"):
        return {
            "url": url,
            "final_url": final_url,
            "status": status,
            "ok": False,
            "label": "❌ No Match",
            "score": 0.0,
            "best_signal": None,
            "signals": [{"source": "error", "value": "", "weight": 0.0, "note": html_text}],
        }
    sigs = extract_author_signals(html_text, final_url)
    score, best, all_sigs = judge_author_match(author_name, sigs)
    label, score_s = classify_conf(score)
    return {
        "url": url,
        "final_url": final_url,
        "status": status,
        "ok": True,
        "label": label,
        "score": score,
        "score_str": score_s,
        "best_signal": (vars(best) if best else None),
        "signals": [vars(s) for s in all_sigs],
    }


# ----------------- UI -----------------

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)

with st.sidebar:
    st.subheader("Settings")
    use_js = st.checkbox(
        "Enable JS rendering (requests-html)",
        value=False,
        help="Helps on some dynamic sites. If not installed, this toggle is ignored."
    )
    st.caption("Signals: JSON-LD > meta > byline > rel=author > title (weak)")

tabs = st.tabs(["🔎 Verify URLs", "📦 JSON Batch"])

# --- Tab 1: Verify URLs
with tabs[0]:
    st.subheader("Author")
    author_name = st.text_input("Author name (required)", placeholder="e.g., Clarissa Chen")

    st.subheader("URLs to check (newline or comma separated)")
    urls_text = st.text_area(
        "Enter URLs",
        placeholder="https://example.com/post-1\nhttps://substack.com/@user/..."
    )
    if st.button("Run Verification", type="primary"):
        if not author_name.strip():
            st.error("Author name is required.")
        else:
            urls = [u.strip() for u in re.split(r"[\n,]+", urls_text or "") if u.strip()]
            if not urls:
                st.error("Please provide at least one URL.")
            else:
                results = []
                prog = st.progress(0.0)
                for i, u in enumerate(urls, 1):
                    res = verify_authorship(author_name, u, use_js=use_js)
                    results.append(res)

                    # Per-URL card
                    st.markdown(f"### {res.get('label','—')} · {res.get('score_str','0.00')}  ")
                    link = res.get("final_url") or res.get("url")
                    st.markdown(f"[{link}]({link})")
                    best = res.get("best_signal")
                    if best:
                        st.caption(f"Best signal: `{best.get('source')}` → {best.get('value')}")
                    with st.expander("Signals"):
                        st.json(res.get("signals", []))
                    st.divider()

                    prog.progress(i / len(urls))
                    time.sleep(0.02)

                # Summary table
                if results:
                    table = []
                    for r in results:
                        table.append({
                            "Label": r.get("label"),
                            "Score": r.get("score_str"),
                            "URL": r.get("final_url") or r.get("url"),
                            "Best Signal": (r.get("best_signal") or {}).get("source") if r.get("best_signal") else "",
                            "Best Value": (r.get("best_signal") or {}).get("value") if r.get("best_signal") else "",
                        })
                    st.subheader("Summary")
                    st.dataframe(table, use_container_width=True)

# --- Tab 2: JSON batch (experts/resources style)
with tabs[1]:
    st.subheader("Paste JSON (list of experts/resources)")
    st.caption("Format example:")
    st.code(
        json.dumps(
            [
                {
                    "author": "Jane Doe",
                    "resources": [
                        {"url": "https://example.com/post-a", "label": "Post A"},
                        {"url": "https://example.com/post-b", "label": "Post B"}
                    ]
                },
                {
                    "author": "John Smith",
                    "resources": [
                        {"url": "https://medium.com/@jsmith/foo", "label": "Medium"}
                    ]
                }
            ],
            indent=2
        ),
        language="json"
    )
    batch_text = st.text_area("Batch JSON", height=220, placeholder="Paste JSON array here…")

    if st.button("Run Batch"):
        try:
            data = json.loads(batch_text or "[]")
            if not isinstance(data, list):
                raise ValueError("Top-level JSON must be a list.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        out_items = []
        total_urls = 0
        authored_urls = 0

        prog = st.progress(0.0)
        idx = 0
        for i, item in enumerate(data):
            author = item.get("author") or ""
            resources = item.get("resources") or []
            res_results = []
            for r in resources:
                url = (r or {}).get("url")
                if not url:
                    continue
                idx += 1
                total_urls += 1
                vr = verify_authorship(author, url, use_js=use_js)
                res_results.append({
                    "url": vr.get("final_url") or vr.get("url"),
                    "label": r.get("label"),
                    "status": vr.get("label"),
                    "score": vr.get("score"),
                    "score_str": vr.get("score_str"),
                    "best_signal": vr.get("best_signal"),
                })
                if vr.get("score", 0.0) >= 0.85:  # count only strong matches
                    authored_urls += 1
                prog.progress(min(1.0, idx / max(1, sum(len((x.get('resources') or [])) for x in data))))

            out_items.append({
                "author": author,
                "resources_checked": len(resources),
                "resources_results": res_results
            })

        st.subheader("Batch Summary")
        st.write(f"Total URLs: {total_urls} · Strong matches (≥0.85): {authored_urls}")

        # Flat table for quick scan
        flat_rows = []
        for it in out_items:
            a = it["author"]
            for rr in it["resources_results"]:
                flat_rows.append({
                    "Author": a,
                    "URL": rr["url"],
                    "Label": rr.get("label"),
                    "Status": rr.get("status"),
                    "Score": rr.get("score_str"),
                    "Best Signal": (rr.get("best_signal") or {}).get("source") if rr.get("best_signal") else "",
                    "Best Value": (rr.get("best_signal") or {}).get("value") if rr.get("best_signal") else ""
                })
        if flat_rows:
            st.dataframe(flat_rows, use_container_width=True)

        with st.expander("Raw results JSON"):
            st.json(out_items)
