# streamlit_app.py
# Generic Directory Scraper → Authorship Verifier
# - Scrapes directory cards for (author, submitted links)
# - Verifies each link via JSON-LD/meta/byline signals
# - Explains a human-readable "Cause" (and excludes About/profile pages)
# Run: streamlit run streamlit_app.py

import pandas as pd
import io
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

# ---------- Optional deps ----------
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

try:
    from playwright.sync_api import sync_playwright
    HAVE_PW = True
except Exception:
    HAVE_PW = False

APP_NAME = "Directory → Authorship Verifier"

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
ABOUT_HINT = re.compile(r"\babout\b|\bteam\b|\babout-us\b|\bour-team\b|\bwho we are\b|\bprofile\b", re.I)
ARTICLE_HREF_HINTS = re.compile(
    r"(article|blog|post|story|insights|news|/p/|/posts?/|/blog/|/stories?/|/insight|/content|/pulse/)",
    re.I,
)

# ---------- Model ----------
@dataclass
class Signal:
    source: str
    value: str
    weight: float
    note: str

# ---------- Utils ----------
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
    out, seen = [], set()
    for u in urls:
        if not u: continue
        u = u.strip()
        if not u: continue
        u, _ = urldefrag(u)
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def same_site(u: str, base: str) -> bool:
    try:
        return urlparse(u).netloc == urlparse(base).netloc
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def fetch(url: str, timeout: int = 25, render_js: bool = False) -> Tuple[int, str, str]:
    try:
        if render_js and HAVE_RHTML:
            session = HTMLSession()
            r = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            try:
                r.html.render(timeout=timeout)
                text = r.html.html or r.text
            except Exception:
                text = r.text
            final_url = str(r.url); status = r.status_code
            session.close()
            return status, text, final_url
        else:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            return r.status_code, r.text, r.url
    except Exception as e:
        return 0, f"__ERROR__ {e}", url

def parse_html(html_text: str) -> BeautifulSoup:
    return BeautifulSoup(html_text, "lxml")

# ---------- JSON-LD safe helpers ----------
def _as_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def _extract_urlish(x):
    if isinstance(x, str): return x.strip()
    if isinstance(x, dict):
        for k in ("@id", "url", "id", "mainEntityOfPage"):
            s = _extract_urlish(x.get(k))
            if s: return s
    if isinstance(x, list):
        for v in x:
            s = _extract_urlish(v)
            if s: return s
    return ""

def _extract_textish(x):
    if isinstance(x, str): return x.strip()
    if isinstance(x, dict):
        for k in ("headline", "name", "title"):
            s = _extract_textish(x.get(k))
            if s: return s
    if isinstance(x, list):
        for v in x:
            s = _extract_textish(v)
            if s: return s
    return ""

def _collect_types(obj):
    out = set()
    if isinstance(obj, dict):
        t = obj.get("@type")
        if isinstance(t, str): out.add(t.lower())
        elif isinstance(t, list): out.update([str(x).lower() for x in t])
    return out

def _flatten_jsonld_entities(blob):
    for item in _as_list(blob):
        if isinstance(item, dict):
            if "@graph" in item and isinstance(item["@graph"], list):
                for g in item["@graph"]:
                    if isinstance(g, dict): yield g
            else:
                yield item

def _extract_authors_from_jsonld(obj):
    names = []
    def collect(val):
        if isinstance(val, str):
            s = val.strip()
            if s: names.append(s)
        elif isinstance(val, dict):
            n = _extract_textish(val.get("name"))
            if n: names.append(n)
        elif isinstance(val, list):
            for v in val: collect(v)
    for key in ("author","creator"):
        if key in obj: collect(obj[key])
    seen=set(); out=[]
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out

def _url_same_page(a, b):
    from urllib.parse import urlparse
    if not a or not b: return False
    pa, pb = urlparse(a), urlparse(b)
    return (pa.netloc.lower(), pa.path.rstrip("/")) == (pb.netloc.lower(), pb.path.rstrip("/"))

# ---------- Page-type classification ----------
def classify_page_type(soup: BeautifulSoup, page_url: str, jsonld_types: List[str]) -> Tuple[str, str]:
    tset = set([t.lower() for t in jsonld_types])
    path = urlparse(page_url).path or ""
    if "person" in tset or "aboutpage" in tset or ABOUT_HINT.search(path):
        h1 = soup.find("h1")
        if h1 and ABOUT_HINT.search(h1.get_text(" ", strip=True) or ""):
            return "about", "H1 indicates About"
        og_type = soup.find("meta", attrs={"property": "og:type"})
        if og_type and (og_type.get("content","").lower() == "profile"):
            return "about", "og:type=profile"
        if "person" in tset and not (tset & {"article","blogposting","newsarticle","videoobject"}):
            return "about", "JSON-LD Person without article type"
    if tset & {"article","blogposting","newsarticle","videoobject"}:
        return "article", "JSON-LD article-like @type"
    h2s = soup.find_all("h2")
    if len(h2s) >= 5:
        return "listing", "Multiple H2s (likely listing)"
    return "other", "No clear page type signals"

# ---------- Signal extraction ----------
def extract_author_signals(html_text: str, page_url: str) -> Tuple[List[Signal], List[str], BeautifulSoup]:
    if html_text.startswith("__ERROR__"):
        return [Signal("error", "", 0.0, html_text)], [], BeautifulSoup("", "lxml")

    soup = parse_html(html_text)
    signals: List[Signal] = []
    jsonld_types_all: List[str] = []

    # Meta tags
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

    # itemprop=author
    for el in soup.select('[itemprop*="author" i]'):
        txt = el.get_text(" ", strip=True)
        if txt:
            signals.append(Signal("itemprop=author", txt, 0.9, "Microdata/HTML author"))

    # JSON-LD
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.text or ""
        if not raw.strip(): continue
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
            is_articleish = bool(types & {"article","blogposting","newsarticle","videoobject"})
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

    # Byline heuristics
    for el in soup.find_all(True, attrs={"class": BYLINE_CLASSES}):
        txt = el.get_text(" ", strip=True)
        if txt:
            signals.append(Signal("byline-class", txt, 0.75, "Byline container"))
            m = BYLINE_REGEX.search(txt)
            if m:
                signals.append(Signal("byline-regex", m.group(1), 0.9, "Byline ‘by …’"))

    # rel=author
    for a in soup.find_all("a", attrs={"rel": re.compile(r"\bauthor\b", re.I)}):
        val = a.get_text(" ", strip=True)
        if val:
            signals.append(Signal("rel=author", val, 0.85, "Rel author link"))

    # Early text "By X"
    body_text = soup.get_text(" ", strip=True)[:1200]
    m = BYLINE_REGEX.search(body_text)
    if m:
        signals.append(Signal("text-byline", m.group(1), 0.8, "Found ‘by …’ near top of page"))

    # Title hint (weak)
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

    # unique JSON-LD types
    types_unique, seen_t = [], set()
    for t in jsonld_types_all:
        t = str(t).lower()
        if t not in seen_t:
            seen_t.add(t); types_unique.append(t)

    return list(dedup.values()), types_unique, soup

def judge_author_match(author_name: str, signals: List[Signal]) -> Tuple[float, Optional[Signal], List[Signal]]:
    author_name = norm(author_name)
    best_s, best_score = None, 0.0
    for s in signals:
        if not s.value: continue
        base = s.weight
        score = sim(author_name, s.value) * base
        if score < 0.4 and s.source in ("title","byline-class") and " by " in norm_name(s.value):
            m = BYLINE_REGEX.search(s.value)
            if m:
                score = max(score, sim(author_name, m.group(1)) * (base + 0.05))
        if score > best_score:
            best_score = score; best_s = s
    return min(1.0, best_score), best_s, signals

def classify_conf(score: float) -> Tuple[str, str]:
    if score >= 0.85: return "✅ Match", f"{score:.2f}"
    if score >= 0.65: return "🟨 Likely", f"{score:.2f}"
    if score >= 0.45: return "🟧 Unclear", f"{score:.2f}"
    return "❌ No Match", f"{score:.2f}"

def explain_cause(author: str, page_url: str, page_type: str, page_type_reason: str,
                  label: str, score: float, best_signal: Optional[Signal], all_signals: List[Signal]) -> str:
    if page_type == "about":
        return "About/profile page — no authored article content"
    if page_type == "listing":
        return "Listing/index page — not a single authored article"
    if not all_signals:
        return "No author signals parsed (meta/byline/JSON-LD not found)"
    if label.startswith("❌"):
        names = []
        for s in all_signals:
            if s.source in ("jsonld","jsonld-weak","meta[name=author]","byline-regex","itemprop=author","rel=author"):
                v = norm(s.value)
                if v and v not in names: names.append(v)
            if len(names) >= 3: break
        if names:
            return f"Author name not found among signals (saw: {', '.join(names)})"
        return "Author name not found (byline, metadata, JSON-LD, title)"
    if best_signal:
        src, val = best_signal.source, best_signal.value
        if src in ("byline-regex","byline-class","text-byline"): return f"Byline matched “{val}”"
        if src.startswith("jsonld"): return f"JSON-LD author “{val}”"
        if src == "meta[name=author]": return f"Meta author tag “{val}”"
        if src.startswith("meta[property="): return f"OpenGraph author “{val}”"
        if src == "itemprop=author": return f"itemprop=author “{val}”"
        if src == "rel=author": return f"rel=author link “{val}”"
        if src == "meta[name=twitter:creator]": return f"Twitter handle “{val}”"
        if src == "title": return "Title hint matched"
    return "Heuristic match"

def verify_authorship(author_name: str, url: str, use_js: bool = False) -> Dict:
    status, html_text, final_url = fetch(url, render_js=use_js)
    if status == 0 or html_text.startswith("__ERROR__"):
        return {
            "url": url, "final_url": final_url, "status": status, "ok": False,
            "label": "❌ No Match", "score": 0.0, "best_signal": None, "signals": [],
            "page_type": "other", "cause": "Fetch error",
        }
    sigs, jsonld_types, soup = extract_author_signals(html_text, final_url)
    score, best, all_sigs = judge_author_match(author_name, sigs)
    label, score_s = classify_conf(score)
    page_type, pt_reason = classify_page_type(soup, final_url, jsonld_types)
    cause = explain_cause(author_name, final_url, page_type, pt_reason, label, score, best, all_sigs)
    return {
        "url": url, "final_url": final_url, "status": status, "ok": True,
        "label": label, "score": score, "score_str": score_s,
        "best_signal": (vars(best) if best else None),
        "signals": [vars(s) for s in all_sigs],
        "page_type": page_type, "cause": cause,
    }

# ---------- Generic directory scraping ----------
def person_like(name: str) -> bool:
    name = norm(name)
    parts = name.split()
    if len(parts) < 2 or len(parts) > 5: return False
    caps = sum(1 for p in parts if re.match(r"^[A-Z][a-z'’.-]+$", p))
    return caps >= max(2, len(parts)-1)

def dedupe_links(links: List[Dict]) -> List[Dict]:
    seen = set(); out = []
    for l in links:
        u = urldefrag(l["url"])[0]
        if u not in seen:
            seen.add(u); out.append(l)
    return out

def nearest_card(node):
    # climb up to a few ancestors looking for a container with a heading + some links
    cur = node
    for _ in range(6):
        if not cur: break
        if (cur.find(["h1","h2","h3","h4"]) is not None) and cur.find("a", href=True):
            return cur
        cur = cur.parent
    return None

def scrape_with_playwright(url: str, mode: str, cfg: Dict) -> List[Dict]:
    """Return [{'author': 'Name', 'links': [{'url':..., 'anchor':...}, ...]}]."""
    results = []
    if not HAVE_PW:
        return results
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=random.choice(USER_AGENTS))
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)

        if mode == "CSS selectors":
            params = {
                "cardSel": cfg.get("card_sel", ""),
                "nameSel": cfg.get("name_sel", ""),
                "linksSel": cfg.get("links_sel", ""),
            }
            data = page.evaluate("""
            (params) => {
              const { cardSel, nameSel, linksSel } = params;
              const OUT = [];
              const cards = document.querySelectorAll(cardSel || "section, article, div");
              cards.forEach(card => {
                const nameEl = nameSel ? card.querySelector(nameSel) : card.querySelector("h1, h2, h3, h4");
                const name = (nameEl && nameEl.innerText || "").trim();
                const links = [];
                (linksSel ? card.querySelectorAll(linksSel) : card.querySelectorAll("a[href]")).forEach(a => {
                  const href = a.getAttribute("href");
                  if (!href) return;
                  links.push({ url: a.href || href, anchor: (a.innerText || "").trim() });
                });
                OUT.push({ name, links });
              });
              return OUT;
            }
            """, params)

        elif mode == "Marker text":
            params = {
                "markerRe": cfg.get("marker_re", "start\\s+exploring"),
                "containerSel": cfg.get("container_sel", "section, article, div"),
            }
            data = page.evaluate("""
            (params) => {
              const { markerRe, containerSel } = params;
              const OUT = [];
              const re = new RegExp(markerRe, 'i');
              const els = Array.from(document.querySelectorAll('*')).filter(el => re.test(el.textContent || ''));
              for (const marker of els) {
                const card = marker.closest(containerSel);
                if (!card) continue;
                const h = card.querySelector("h1, h2, h3, h4");
                const name = (h && h.innerText || '').trim();
                const links = [];
                const as = Array.from(card.querySelectorAll('a[href]'));
                for (const a of as) {
                  if (marker.compareDocumentPosition(a) & Node.DOCUMENT_POSITION_FOLLOWING) {
                    links.push({ url: a.href, anchor: (a.innerText || '').trim() });
                  }
                }
                OUT.push({ name, links });
              }
              return OUT;
            }
            """, params)

        else:  # Heuristic
            data = page.evaluate("""
            () => {
              const OUT = [];
              const cards = document.querySelectorAll("section, article, div");
              cards.forEach(card => {
                const h = card.querySelector("h1, h2, h3, h4");
                const name = (h && h.innerText || '').trim();
                const links = [];
                card.querySelectorAll("a[href]").forEach(a => {
                  links.push({ url: a.href, anchor: (a.innerText || '').trim() });
                });
                OUT.push({ name, links });
              });
              return OUT;
            }
            """)

        browser.close()

    # Post-filter & clean
    out = []
    for block in data or []:
        name = norm(block.get("name"))
        if not person_like(name):
            continue
        seen, links = set(), []
        for l in block.get("links") or []:
            href = norm(l.get("url"))
            if not href or href in seen:
                continue
            seen.add(href)
            links.append({"url": href, "anchor": norm(l.get("anchor"))})
        if links:
            out.append({"author": name, "links": links})
    return out

def scrape_with_bs4(url: str, mode: str, cfg: Dict) -> List[Dict]:
    status, html, final = fetch(url, render_js=False)
    if status <= 0 or html.startswith("__ERROR__"):
        return []
    soup = parse_html(html)
    out = []

    if mode == "CSS selectors":
        card_sel = cfg.get("card_sel") or "section, article, div"
        name_sel = cfg.get("name_sel") or "h1, h2, h3, h4"
        links_sel = cfg.get("links_sel") or "a[href]"

        for card in soup.select(card_sel):
            name_el = card.select_one(name_sel)
            name = name_el.get_text(" ", strip=True) if name_el else ""
            if not person_like(name):
                continue
            links = []
            for a in card.select(links_sel):
                href = a.get("href")
                if not href:
                    continue
                links.append({"url": urljoin(final, href.strip()), "anchor": a.get_text(" ", strip=True)})
            links = dedupe_links(links)
            if links:
                out.append({"author": name, "links": links})
        return out

    if mode == "Marker text":
        marker_re = re.compile(cfg.get("marker_re") or r"start\s+exploring", re.I)
        for text_node in soup.find_all(string=marker_re):
            marker = text_node.parent
            card = nearest_card(marker)
            if not card:
                continue
            # name
            h = card.find(["h1","h2","h3","h4"])
            name = h.get_text(" ", strip=True) if h else ""
            if not person_like(name):
                continue
            # links after marker inside card (best-effort using sourceline if present)
            links = []
            for a in card.find_all("a", href=True):
                try:
                    if hasattr(a, "sourceline") and hasattr(marker, "sourceline"):
                        if a.sourceline is not None and marker.sourceline is not None and a.sourceline < marker.sourceline:
                            continue
                except Exception:
                    pass
                links.append({"url": urljoin(final, a["href"].strip()), "anchor": a.get_text(" ", strip=True)})
            links = dedupe_links(links)
            if links:
                out.append({"author": name, "links": links})
        if out:
            return out

    # Heuristic fallback: find cards with person-like headings and collect links inside
    for card in soup.find_all(["section","article","div"]):
        h = card.find(["h1","h2","h3","h4"])
        name = h.get_text(" ", strip=True) if h else ""
        if not person_like(name):
            continue
        links = [{"url": urljoin(final, a["href"].strip()), "anchor": a.get_text(" ", strip=True)}
                 for a in card.find_all("a", href=True)]
        links = dedupe_links(links)
        if links:
            out.append({"author": name, "links": links})
    return out

def scrape_directory(url: str, mode: str, cfg: Dict, prefer_playwright: bool) -> List[Dict]:
    if prefer_playwright and HAVE_PW:
        data = scrape_with_playwright(url, mode, cfg)
        if data:
            return data
    return scrape_with_bs4(url, mode, cfg)

# ---------- UI ----------
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)

with st.sidebar:
    st.subheader("Verification settings")
    use_js = st.checkbox("Enable JS rendering for verification (requests-html)", value=False)
    st.caption("Used only when fetching target article URLs.")
    st.subheader("Scraper settings")
    prefer_pw = st.checkbox("Prefer Playwright for scraping", value=True)
    same_site_only = st.checkbox("Keep only same-site links", value=False)
    include_pat = st.text_input("Include REGEX (href filter)", value=r"")
    exclude_pat = st.text_input("Exclude REGEX (href filter)", value=r"")
    max_links_per_author = st.number_input("Max links per author", min_value=1, max_value=50, value=12)

st.subheader("1) Directory URLs (one per line)")
dirs_text = st.text_area("Paste listing pages", placeholder="https://www.mlforseo.com/experts/\nhttps://example.com/directory", height=100)

st.subheader("2) Scraping mode")
mode = st.radio("Choose one", ["CSS selectors", "Marker text", "Heuristic"], horizontal=True)

cfg = {}
if mode == "CSS selectors":
    col1, col2 = st.columns(2)
    with col1:
        cfg["card_sel"] = st.text_input("Card selector", "section, article, div")
        cfg["name_sel"] = st.text_input("Author name selector (within card)", "h1, h2, h3, h4")
    with col2:
        cfg["links_sel"] = st.text_input("Link selector (within card)", "a[href]")
    st.caption("Tip: target the exact sub-sections that contain the submitted links if possible.")
elif mode == "Marker text":
    cfg["marker_re"] = st.text_input("Marker text (regex)", r"start\s+exploring")
    cfg["container_sel"] = st.text_input("Card container selector (closest ancestor)", "section, article, div")
else:
    st.caption("Heuristic looks for cards with person-like headings and collects links inside each card.")

run_scrape = st.button("Scrape directory pages", type="primary")

if run_scrape:
    urls = clean_urls(re.split(r"[\n,]+", dirs_text or ""))
    if not urls:
        st.error("Provide at least one directory URL.")
    else:
        all_cards = []
        for u in urls:
            st.write(f"Scraping: {u}")
            data = scrape_directory(u, mode, cfg, prefer_playwright=prefer_pw)
            # filters: same-site/include/exclude; cap per author
            base = u
            inc = re.compile(include_pat, re.I) if include_pat else None
            exc = re.compile(exclude_pat, re.I) if exclude_pat else None
            for item in data:
                links = []
                for l in item["links"]:
                    url = l["url"]
                    if same_site_only and not same_site(url, base):
                        continue
                    if inc and not inc.search(url): continue
                    if exc and exc.search(url): continue
                    links.append(l)
                    if len(links) >= max_links_per_author:
                        break
                if links:
                    all_cards.append({"author": item["author"], "links": links})
        if not all_cards:
            st.warning("No author/link groups found. Adjust selectors/mode or try enabling Playwright.")
        else:
            st.success(f"Found {len(all_cards)} author groups.")
            # preview
            preview = []
            for it in all_cards[:50]:
                preview.append({
                    "Author": it["author"],
                    "Links (#)": len(it["links"]),
                    "Examples": ", ".join([l["url"] for l in it["links"][:3]])
                })
            st.dataframe(preview, use_container_width=True)
            st.session_state["_scraped_groups"] = all_cards

st.subheader("3) Verify scraped links")
if st.button("Verify now", type="primary"):
    groups = st.session_state.get("_scraped_groups") or []
    if not groups:
        st.warning("Nothing scraped yet.")
    else:
        total = sum(len(g["links"]) for g in groups) or 1
        done = 0
        prog = st.progress(0.0)
        rows = []

        for g in groups:
            author = g["author"]
            st.markdown(f"### {author}")
            for l in g["links"]:
                url = l["url"]
                res = verify_authorship(author, url, use_js=use_js)
                done += 1; prog.progress(done/total)
                st.markdown(f"**{res.get('label','—')}** · {res.get('score_str','0.00')}  \n[{res.get('final_url') or res.get('url')}]({res.get('final_url') or res.get('url')})")
                st.write(f"**Cause:** {res.get('cause','')}")
                best = res.get("best_signal")
                if best:
                    st.caption(f"Best signal: `{best.get('source')}` → {best.get('value')}")
                with st.expander("Signals"):
                    st.json(res.get("signals", []))
                st.divider()
                rows.append({
                    "Author": author,
                    "URL": res.get("final_url") or res.get("url"),
                    "Status": res.get("label"),
                    "Score": res.get("score_str"),
                    "Cause": res.get("cause",""),
                    "Best Signal": (best or {}).get("source") if best else "",
                    "Best Value": (best or {}).get("value") if best else "",
                })

        st.subheader("Summary table")
if rows:
    # Full results table
    df = pd.DataFrame(
        rows,
        columns=["Author", "URL", "Status", "Score", "Cause", "Best Signal", "Best Value"]
    )
    st.dataframe(df, use_container_width=True)

    # Per-author rollup
    st.subheader("Per-author rollup")
    rollup = (
        df.assign(Count=1)
          .pivot_table(index="Author", columns="Status", values="Count", aggfunc="sum", fill_value=0)
          .reset_index()
    )
    st.dataframe(rollup, use_container_width=True)

    # Downloads: JSON + CSV (+ Excel if xlsxwriter is available)
    st.download_button(
        "Download results.json",
        data=json.dumps(rows, indent=2).encode("utf-8"),
        file_name="directory_authorship_results.json",
        mime="application/json",
    )

    st.download_button(
        "Download results.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="directory_authorship_results.csv",
        mime="text/csv",
    )

    try:
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)
            rollup.to_excel(writer, sheet_name="Rollup", index=False)
        st.download_button(
            "Download results.xlsx",
            data=xlsx_buf.getvalue(),
            file_name="directory_authorship_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.caption("Tip: install `xlsxwriter` to enable Excel export (pip install xlsxwriter).")
