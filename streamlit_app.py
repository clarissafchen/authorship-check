# streamlit_app.py
# Authorship Verifier — directory crawl + explicit "cause" + robust signals
# Run: streamlit run streamlit_app.py

import re
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin, urldefrag

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
ABOUT_HINT = re.compile(r"\babout\b|\bteam\b|\babout-us\b|\bour-team\b|\bwho we are\b", re.I)

ARTICLE_HREF_HINTS = re.compile(
    r"(article|blog|post|story|insights|news|/p/|/posts?/|/blog/|/stories?/|/insight|/content|/pulse/)",
    re.I,
)

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

def clean_urls(urls: List[str]) -> List[str]:
    out = []
    seen = set()
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        u, _ = urldefrag(u)
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

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

# ----------------- JSON-LD safe helpers -----------------

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

# ----------------- Page-type classification -----------------

def classify_page_type(soup: BeautifulSoup, page_url: str, jsonld_types: List[str]) -> Tuple[str, str]:
    """
    Returns (page_type, reason):
      - 'article' for Article/BlogPosting/NewsArticle/VideoObject-ish
      - 'about' for Person/AboutPage/profile pages
      - 'listing' for home/list pages (heuristic)
      - 'other' default
    """
    tset = set([t.lower() for t in jsonld_types])
    path = urlparse(page_url).path or ""

    # About/Profile signals
    if "person" in tset or "aboutpage" in tset or ABOUT_HINT.search(path):
        h1 = soup.find("h1")
        if h1 and ABOUT_HINT.search(h1.get_text(" ", strip=True) or ""):
            return "about", "H1 indicates About"
        if ABOUT_HINT.search(path):
            return "about", "URL indicates About"
        # Some profile pages expose og:type=profile
        og_type = soup.find("meta", attrs={"property": "og:type"})
        if og_type and (og_type.get("content","").lower() == "profile"):
            return "about", "og:type=profile"
        # JSON-LD Person without article types
        if "person" in tset and not (tset & {"article","blogposting","newsarticle","videoobject"}):
            return "about", "JSON-LD Person without article type"

    # Article-like signals
    if tset & {"article", "blogposting", "newsarticle", "videoobject"}:
        return "article", "JSON-LD article-like @type"

    # Listing heuristics
    # Many h2 cards linking to posts + no obvious author signals
    h2s = soup.find_all("h2")
    if len(h2s) >= 5:
        return "listing", "Multiple H2s (likely listing)"

    return "other", "No clear page type signals"

# ----------------- Signal extraction -----------------

def extract_author_signals(html_text: str, page_url: str) -> Tuple[List[Signal], List[str], BeautifulSoup]:
    if html_text.startswith("__ERROR__"):
        return [Signal("error", "", 0.0, html_text)], [], BeautifulSoup("", "lxml")

    soup = parse_html(html_text)
    signals: List[Signal] = []
    jsonld_types_all: List[str] = []

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

    # 2) itemprop=author
    for el in soup.select('[itemprop*="author" i]'):
        txt = el.get_text(" ", strip=True)
        if txt:
            signals.append(Signal("itemprop=author", txt, 0.9, "Microdata/HTML author"))

    # 3) JSON-LD (robust)
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.text or ""
        if not raw.strip():
            continue
        try:
            blob = json.loads(raw)
        except Exception:
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
            jsonld_types_all.extend(list(types))
            is_articleish = bool(types & {"article", "blogposting", "newsarticle", "videoobject"})
            obj_url = _extract_urlish(obj.get("url")) or _extract_urlish(obj.get("mainEntityOfPage"))
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

    # 4) Byline heuristics
    for el in soup.find_all(True, attrs={"class": BYLINE_CLASSES}):
        txt = el.get_text(" ", strip=True)
        if txt:
            signals.append(Signal("byline-class", txt, 0.75, "Byline container"))
            m = BYLINE_REGEX.search(txt)
            if m:
                signals.append(Signal("byline-regex", m.group(1), 0.9, "Byline ‘by …’"))

    # 5) rel=author
    for a in soup.find_all("a", attrs={"rel": re.compile(r"\bauthor\b", re.I)}):
        val = a.get_text(" ", strip=True)
        if val:
            signals.append(Signal("rel=author", val, 0.85, "Rel author link"))

    # 6) Generic early-text scan for "By X" near top
    body_text = soup.get_text(" ", strip=True)[:1200]
    m = BYLINE_REGEX.search(body_text)
    if m:
        signals.append(Signal("text-byline", m.group(1), 0.8, "Found ‘by …’ near top of page"))

    # 7) Title hint (weak)
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        if title:
            signals.append(Signal("title", title, 0.25, "Weak heuristic from <title>"))

    # Dedup
    dedup = {}
    for s in signals:
        key = (s.source, norm_name(s.value))
        if key not in dedup or s.weight > dedup[key].weight:
            dedup[key] = s

    # Consolidated list of JSON-LD types (unique)
    all_types = []
    seen_t = set()
    for t in jsonld_types_all:
        t = str(t).lower()
        if t not in seen_t:
            seen_t.add(t)
            all_types.append(t)

    return list(dedup.values()), all_types, soup

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

# ----------------- Cause/diagnostics -----------------

def explain_cause(author: str, page_url: str, page_type: str, page_type_reason: str,
                  label: str, score: float, best_signal: Optional[Signal], all_signals: List[Signal]) -> str:
    # About/profile pages
    if page_type == "about":
        return "About/profile page — no authored article content"

    # Listing
    if page_type == "listing":
        return "Listing/index page — not a single authored article"

    # No signals at all
    if not all_signals:
        return "No author signals parsed (meta/byline/JSON-LD not found)"

    # If No Match, surface top author-like names we DID find (up to 3)
    if label.startswith("❌"):
        names = []
        for s in all_signals:
            if s.source in ("jsonld", "jsonld-weak", "meta[name=author]", "byline-regex", "itemprop=author", "rel=author"):
                v = norm(s.value)
                if v and v not in names:
                    names.append(v)
            if len(names) >= 3:
                break
        if names:
            return f"Author name not found among signals (saw: {', '.join(names)})"
        return "Author name not found (byline, metadata, JSON-LD, title)"

    # For matches/likely/unclear: explain best signal
    if best_signal:
        src = best_signal.source
        val = best_signal.value
        if src in ("byline-regex", "byline-class", "text-byline"):
            return f"Byline matched “{val}”"
        if src.startswith("jsonld"):
            return f"JSON-LD author “{val}”"
        if src == "meta[name=author]":
            return f"Meta author tag “{val}”"
        if src.startswith("meta[property="):
            return f"OpenGraph author “{val}”"
        if src == "itemprop=author":
            return f"itemprop=author “{val}”"
        if src == "rel=author":
            return f"rel=author link “{val}”"
        if src == "meta[name=twitter:creator]":
            return f"Twitter handle “{val}”"
        if src == "title":
            return "Title hint matched"
    return "Heuristic match"

# ----------------- Verification -----------------

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
            "signals": [],
            "page_type": "other",
            "cause": "Fetch error",
        }
    sigs, jsonld_types, soup = extract_author_signals(html_text, final_url)
    score, best, all_sigs = judge_author_match(author_name, sigs)
    label, score_s = classify_conf(score)

    # classify page
    page_type, pt_reason = classify_page_type(soup, final_url, jsonld_types)

    cause = explain_cause(author_name, final_url, page_type, pt_reason, label, score, best, all_sigs)

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
        "page_type": page_type,
        "cause": cause,
    }

# ----------------- Directory crawl helpers -----------------

def absolute_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:") or href.startswith("tel:") or href.startswith("#"):
            continue
        links.append(urljoin(base_url, href))
    return clean_urls(links)

def filter_articleish(urls: List[str], base_url: str, include_re: Optional[str], exclude_re: Optional[str]) -> List[str]:
    out = []
    inc = re.compile(include_re, re.I) if include_re else None
    exc = re.compile(exclude_re, re.I) if exclude_re else None
    base_netloc = urlparse(base_url).netloc

    for u in urls:
        if urlparse(u).netloc and urlparse(u).netloc != base_netloc:
            continue
        if inc and not inc.search(u):
            continue
        if exc and exc.search(u):
            continue
        if inc is None:
            if not ARTICLE_HREF_HINTS.search(u):
                continue
        out.append(u)
    return out

def extract_links_with_selector(soup: BeautifulSoup, base_url: str, selector: str) -> List[str]:
    out = []
    for el in soup.select(selector):
        if el.name == "a" and el.has_attr("href"):
            out.append(urljoin(base_url, el["href"]))
        else:
            a = el.find("a", href=True)
            if a:
                out.append(urljoin(base_url, a["href"]))
    return clean_urls(out)

# ----------------- UI -----------------

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)

with st.sidebar:
    st.subheader("Settings")
    use_js = st.checkbox(
        "Enable JS rendering (requests-html)",
        value=False,
        help="Helps on dynamic pages. If not installed, this toggle is ignored."
    )
    st.caption("Signals: JSON-LD > meta > byline/itemprop/rel=author > title (weak)")

tabs = st.tabs(["🕷️ Crawl & Verify Directory", "🔎 Verify URLs", "📦 JSON Batch"])

# --- Tab 1: Crawl & Verify Directory
with tabs[0]:
    st.subheader("1) Directory / Listing URL")
    colA, colB = st.columns([2, 1])
    with colA:
        dir_url = st.text_input("Directory URL", placeholder="https://example.com/blog/")
    with colB:
        crawl_limit = st.number_input("Max articles", min_value=1, max_value=500, value=50, step=5)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        custom_selector = st.text_input(
            "Optional CSS selector for article links",
            placeholder="e.g., .posts-list article a, .card a"
        )
    with col2:
        include_pat = st.text_input(
            "Optional include REGEX (href filter)",
            placeholder=r"(blog|post|article|pulse)"
        )
    with col3:
        exclude_pat = st.text_input(
            "Optional exclude REGEX (href filter)",
            placeholder=r"(\?replytocom=|/tag/|/category/)"
        )

    st.subheader("2) Author to Verify")
    author_name = st.text_input("Author name (required)", placeholder="e.g., Chai Fisher", key="crawl_author")

    go = st.button("Crawl & Extract Links")
    if go and dir_url:
        with st.spinner("Fetching directory..."):
            status, html_text, info_url = fetch(dir_url, render_js=use_js)
        if status <= 0 or html_text.startswith("__ERROR__"):
            st.error(f"Fetch error ({status}). {html_text}")
        else:
            soup = parse_html(html_text)
            if custom_selector.strip():
                raw_links = extract_links_with_selector(soup, info_url, custom_selector.strip())
            else:
                raw_links = absolute_links(soup, info_url)
                raw_links = filter_articleish(raw_links, info_url, include_pat or None, exclude_pat or None)

            if not raw_links:
                st.warning("No candidate article links found. Provide a CSS selector or adjust include/exclude patterns.")
            else:
                links = raw_links[:crawl_limit]
                st.success(f"Found {len(links)} article candidates (showing up to {crawl_limit}).")
                st.dataframe({"URL": links})
                if author_name.strip():
                    run_bulk = st.button("Verify Authorship for These Links")
                    if run_bulk:
                        results = []
                        prog = st.progress(0.0)
                        for i, u in enumerate(links, 1):
                            results.append(verify_authorship(author_name, u, use_js))
                            prog.progress(i / len(links))
                            time.sleep(0.02)
                        # Summary table with CAUSE
                        table = []
                        for r in results:
                            table.append({
                                "Label": r.get("label"),
                                "Score": r.get("score_str"),
                                "URL": r.get("final_url") or r.get("url"),
                                "Best Signal": (r.get("best_signal") or {}).get("source") if r.get("best_signal") else "",
                                "Best Value": (r.get("best_signal") or {}).get("value") if r.get("best_signal") else "",
                                "Cause": r.get("cause",""),
                            })
                        st.subheader("Results")
                        st.dataframe(table, use_container_width=True)
                        with st.expander("Diagnostics (per URL)"):
                            st.json(results)
                else:
                    st.info("Enter an author name above to run verification on the found links.")

# --- Tab 2: Verify URLs (single author)
with tabs[1]:
    st.subheader("Author")
    author_single = st.text_input("Author name (required)", placeholder="e.g., Chai Fisher", key="author_single")

    st.subheader("URLs to check (newline or comma separated)")
    urls_text = st.text_area(
        "Enter URLs",
        placeholder="https://example.com/post-1\nhttps://substack.com/@user/..."
    )
    if st.button("Run Verification", type="primary"):
        if not author_single.strip():
            st.error("Author name is required.")
        else:
            urls = clean_urls([u.strip() for u in re.split(r"[\n,]+", urls_text or "") if u.strip()])
            if not urls:
                st.error("Please provide at least one URL.")
            else:
                results = []
                prog = st.progress(0.0)
                for i, u in enumerate(urls, 1):
                    res = verify_authorship(author_single, u, use_js=use_js)
                    results.append(res)

                    # Per-URL card
                    st.markdown(f"### {res.get('label','—')} · {res.get('score_str','0.00')}  ")
                    link = res.get("final_url") or res.get("url")
                    st.markdown(f"[{link}]({link})")
                    st.write(f"**Cause:** {res.get('cause','')}")
                    best = res.get("best_signal")
                    if best:
                        st.caption(f"Best signal: `{best.get('source')}` → {best.get('value')}")
                    with st.expander("Signals"):
                        st.json(res.get("signals", []))
                    st.divider()

                    prog.progress(i / len(urls))
                    time.sleep(0.02)

                # Summary table (includes CAUSE)
                if results:
                    table = []
                    for r in results:
                        table.append({
                            "Label": r.get("label"),
                            "Score": r.get("score_str"),
                            "URL": r.get("final_url") or r.get("url"),
                            "Best Signal": (r.get("best_signal") or {}).get("source") if r.get("best_signal") else "",
                            "Best Value": (r.get("best_signal") or {}).get("value") if r.get("best_signal") else "",
                            "Cause": r.get("cause",""),
                        })
                    st.subheader("Summary")
                    st.dataframe(table, use_container_width=True)

# --- Tab 3: JSON batch (experts/resources style)
with tabs[2]:
    st.subheader("Paste JSON (list of experts/resources)")
    st.caption("Format:")
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
        total_expected = sum(len((x.get("resources") or [])) for x in data) or 1

        prog = st.progress(0.0)
        idx = 0
        for item in data:
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
                    "cause": vr.get("cause"),
                })
                if vr.get("score", 0.0) >= 0.85:
                    authored_urls += 1
                prog.progress(min(1.0, idx / total_expected))
                time.sleep(0.01)

            out_items.append({
                "author": author,
                "resources_checked": len(resources),
                "resources_results": res_results
            })

        st.subheader("Batch Summary")
        st.write(f"Total URLs: {total_urls} · Strong matches (≥0.85): {authored_urls}")

        # Flat table for quick scan (includes CAUSE)
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
                    "Best Value": (rr.get("best_signal") or {}).get("value") if rr.get("best_signal") else "",
                    "Cause": rr.get("cause",""),
                })
        if flat_rows:
            st.dataframe(flat_rows, use_container_width=True)

        with st.expander("Raw results JSON"):
            st.json(out_items)
