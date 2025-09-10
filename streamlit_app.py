# Streamlit app:
#   - Tab 1: Scrape https://www.mlforseo.com/experts/ grid (requests + BeautifulSoup)
#   - Tab 2: Verify authorship of submitted content URLs against a provided author name
#
# Notes:
# - Author check gathers evidence from JSON-LD, meta tags, visible bylines, domain adapters
# - LinkedIn Pulse often blocks bots; we add slug heuristics + JSON-LD regex fallback
# - Matching uses normalized tokens + fuzzy ratio (difflib) to classify MATCH/POSSIBLE/NO_MATCH

# streamlit_app.py
import json
import re
import time
import unicodedata
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

APP_TITLE = "MLforSEO Tools"

# ---------- HTTP ----------
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

def http_get(url: str, timeout: int = 20) -> requests.Response:
    """Simple GET with friendly headers."""
    return requests.get(
        url,
        headers={
            "User-Agent": UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        },
        timeout=timeout,
        allow_redirects=True,
    )

# ---------- Text / Names ----------
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s or "")
        if unicodedata.category(c) != "Mn"
    )

def norm_text(s: str) -> str:
    s = strip_accents(s or "")
    s = re.sub(r"[\u2010-\u2015\-–—]", " ", s)         # dashes → space
    s = re.sub(r"[^a-zA-Z\s]+", " ", s)               # keep letters/spaces
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def name_tokens(name: str) -> list[str]:
    return [t for t in norm_text(name).split() if len(t) > 1]

def token_similarity(a_tokens: list[str], b_tokens: list[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    sa, sb = set(a_tokens), set(b_tokens)
    return len(sa & sb) / len(sa | sb)

def names_match(expected: str, candidate: str) -> tuple[bool, float]:
    """Loose match tolerant of middle names and reordering."""
    te = name_tokens(expected)
    tc = name_tokens(candidate)
    if not te or not tc:
        return False, 0.0

    # Strong conditions
    if set(te).issubset(set(tc)) or set(tc).issubset(set(te)):
        conf = 0.95 if (te and tc and te[-1] == tc[-1]) else 0.85
        return True, conf

    sim = token_similarity(te, tc)
    # last-name + first initial helps
    last_ok = te[-1] == tc[-1]
    first_initial_ok = te[0][0] == tc[0][0]
    if sim >= 0.67 or (last_ok and (first_initial_ok or te[0] == tc[0])):
        return True, max(sim, 0.75 if last_ok else 0.7)

    return False, sim

# ---------- Author extraction ----------
BYLINE_SELECTORS = [
    '[itemprop="author"]',
    '[rel="author"]',
    '.byline', '.by-line', '.entry-author', '.post-author',
    '.article-author', '.article__author', '.author-name', '.author',
    '.c-article-author', '.amp-author', '.meta-author',
    '[data-testid*="author"]', '[data-test*="author"]',
    '[class*="author"]', '[class*="byline"]',
    # creator-ish selectors help for video platforms like Loom
    '[class*="creator"]', '[data-testid*="creator"]', '[data-test*="creator"]',
]

def _walk_json_for_names(obj) -> list[str]:
    out = []
    keys_of_interest = ("author", "creator", "accountablePerson", "uploader", "byline")
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.lower()
            if any(x in lk for x in keys_of_interest):
                # direct string
                if isinstance(v, str) and 1 < len(v) < 120:
                    out.append(v)
                # nested objects
                elif isinstance(v, dict):
                    nm = v.get("name") or v.get("alternateName") or v.get("title") or ""
                    if 1 < len(nm) < 120:
                        out.append(nm)
                # lists
                elif isinstance(v, list):
                    for i in v:
                        if isinstance(i, str) and 1 < len(i) < 120:
                            out.append(i)
                        elif isinstance(i, dict):
                            nm = i.get("name") or i.get("alternateName") or i.get("title") or ""
                            if 1 < len(nm) < 120:
                                out.append(nm)
            # keep walking
            out.extend(_walk_json_for_names(v))
    elif isinstance(obj, list):
        for i in obj:
            out.extend(_walk_json_for_names(i))
    return out

def extract_authors_jsonld(soup: BeautifulSoup) -> list[str]:
    names = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or tag.text)
        except Exception:
            continue
        names.extend(_walk_json_for_names(data))
    return names

def extract_authors_meta(soup: BeautifulSoup) -> list[str]:
    names = []
    for sel_attr, sel_val in (("name", "author"), ("property", "article:author")):
        for m in soup.select(f'meta[{sel_attr}="{sel_val}"]'):
            content = (m.get("content") or "").strip()
            if content and len(content) < 120:
                names.append(content)
    return names

def extract_authors_bylines(soup: BeautifulSoup) -> list[str]:
    found = []
    for sel in BYLINE_SELECTORS:
        for el in soup.select(sel):
            txt = " ".join(el.stripped_strings)
            if 2 < len(txt) < 160:
                found.append(txt)
    cleaned = []
    for t in found:
        t2 = re.sub(r"^(?:by|author)\s*[:\-–—]\s*", "", t, flags=re.I)
        t2 = re.split(r"[|•·,;/]| and |\s{2,}", t2)[0].strip()
        if t2:
            cleaned.append(t2)
    return cleaned

def guess_author_from_url(url: str) -> str:
    """Heuristic: extract names from slugs when sites hide bylines behind auth walls (e.g., LinkedIn Pulse)."""
    path = urlparse(url).path.lower()

    # LinkedIn Pulse: .../pulse/<title>-firstname-lastname-<hash>
    m = re.search(r"/pulse/[^/]*-([a-z]+)-([a-z]+)-[a-z0-9]+/?$", path)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    # Generic last-two alpha tokens in slug
    slug = path.strip("/").split("/")[-1] if path.strip("/") else ""
    slug = re.sub(r"\.(html|htm)$", "", slug)
    tokens = [t for t in slug.split("-") if t.isalpha()]
    if len(tokens) >= 2:
        return f"{tokens[-2]} {tokens[-1]}"

    return ""

def extract_authors_site_specific(url: str, soup: BeautifulSoup) -> list[str]:
    host = urlparse(url).netloc.lower()
    out = []

    # Loom: try Next.js data dump for "creator"/"owner" fields, plus visible byline
    if "loom.com" in host:
        data_tag = soup.find("script", id="__NEXT_DATA__")
        if data_tag and (data_tag.string or data_tag.text):
            try:
                data = json.loads(data_tag.string or data_tag.text)
                out.extend(_walk_json_for_names(data))
            except Exception:
                pass

    # Women in Tech SEO: explicit "Author:" label on page
    if "womenintechseo.com" in host:
        for el in soup.find_all(string=re.compile(r"Author:", re.I)):
            line = el.parent.get_text(" ", strip=True)
            nm = re.sub(r".*Author:\s*", "", line, flags=re.I).strip()
            if 1 < len(nm) < 120:
                out.append(nm)

    # Add more site adapters here as needed.

    return out

def detect_authors(url: str) -> dict:
    """Fetch a URL and extract candidate author names using multiple strategies."""
    try:
        r = http_get(url)
        status = r.status_code
        html = r.text if status == 200 else ""
    except Exception as e:
        return {
            "url": url,
            "http_status": None,
            "error": f"{type(e).__name__}: {e}",
            "authors": [],
            "methods": [],
        }

    if not html:
        return {"url": url, "http_status": status, "authors": [], "methods": []}

    soup = BeautifulSoup(html, "lxml")
    authors, methods = [], []

    # JSON-LD
    a = extract_authors_jsonld(soup)
    if a:
        authors.extend(a)
        methods.extend(["jsonld"] * len(a))

    # Meta tags
    a = extract_authors_meta(soup)
    if a:
        authors.extend(a)
        methods.extend(["meta"] * len(a))

    # Visible bylines
    a = extract_authors_bylines(soup)
    if a:
        authors.extend(a)
        methods.extend(["byline"] * len(a))

    # Site-specific
    a = extract_authors_site_specific(url, soup)
    if a:
        authors.extend(a)
        methods.extend(["site"] * len(a))

    # URL slug heuristic at the end (most fragile)
    slug_guess = guess_author_from_url(url)
    if slug_guess:
        authors.insert(0, slug_guess)
        methods.insert(0, "slug")

    # Deduplicate (preserve order)
    seen = set()
    uniq, uniq_methods = [], []
    for nm, m in zip(authors, methods):
        key = norm_text(nm)
        if key and key not in seen:
            seen.add(key)
            uniq.append(nm.strip())
            uniq_methods.append(m)

    return {
        "url": url,
        "http_status": status,
        "authors": uniq,
        "methods": uniq_methods,
    }

def evaluate_authorship(url: str, expected_author: str) -> dict:
    det = detect_authors(url)
    best_match = {"candidate": "", "confidence": 0.0, "matched": False, "method": ""}

    for cand, method in zip(det["authors"], det.get("methods", [])):
        ok, score = names_match(expected_author, cand)
        if ok and score > best_match["confidence"]:
            best_match = {
                "candidate": cand,
                "confidence": round(float(score), 3),
                "matched": True,
                "method": method,
            }

    # If nothing matched, still report best similarity candidate
    if not best_match["matched"] and det["authors"]:
        # choose candidate with highest similarity
        top = sorted(
            ((cand, token_similarity(name_tokens(expected_author), name_tokens(cand)), m)
             for cand, m in zip(det["authors"], det.get("methods", []))),
            key=lambda x: x[1],
            reverse=True,
        )[0]
        best_match = {
            "candidate": top[0],
            "confidence": round(float(top[1]), 3),
            "matched": False,
            "method": top[2],
        }

    return {
        "url": url,
        "http_status": det.get("http_status"),
        "expected_author": expected_author,
        "authors_detected": det.get("authors"),
        "methods": det.get("methods"),
        "author_match": best_match["matched"],
        "matched_author": best_match["candidate"],
        "confidence": best_match["confidence"],
        "evidence_method": best_match["method"],
        "error": det.get("error"),
    }

# ---------- Experts grid scraper ----------
def parse_experts_grid(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    cards = soup.select("div.experts-grid article.expert-card")
    experts = []
    for card in cards:
        # NOTE: "author" (not "name") per request
        author = (card.select_one("h3.name a") or card.select_one("h3.name"))
        author = (author.get_text(strip=True) if author else "").strip()

        site = card.select_one(".icons a.icon-link.site")
        site_url = site.get("href").strip() if site and site.has_attr("href") else ""

        img = card.select_one(".photo-wrap img")
        headshot = img.get("src").strip() if img and img.has_attr("src") else ""

        reason = card.select_one("p.reason")
        reason = reason.get_text(" ", strip=True) if reason else ""

        chips = [c.get_text(strip=True) for c in card.select(".chips-row .chip")]

        resources = []
        for a in card.select(".resources-list a.res-pill"):
            href = a.get("href")
            if not href:
                continue
            href = href.strip()
            label = a.select_one(".res-text")
            label_txt = label.get_text(strip=True) if label else urlparse(href).netloc
            resources.append({"url": href, "label": label_txt})

        experts.append(
            {
                "author": author,
                "website": site_url,
                "headshot": headshot,
                "chips": chips,
                "reason": reason,
                "resources": resources,
            }
        )
    return experts

def scrape_experts(url: str) -> tuple[list[dict], dict]:
    diags = {"http": {}, "parse": {}}
    t0 = time.time()
    r = http_get(url)
    diags["http"] = {
        "requested_url": url,
        "final_url": r.url,
        "status_code": r.status_code,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "length": len(r.text or ""),
        "engine": "requests",
    }
    experts = []
    if r.status_code == 200:
        experts = parse_experts_grid(r.text, r.url)
        diags["parse"] = {
            "parser": "mlforseo_grid_css",
            "cards_found": len(experts),
            "experts_parsed": len(experts),
        }
    else:
        diags["parse"] = {"parser": "mlforseo_grid_css", "error": f"HTTP {r.status_code}"}
    return experts, diags

def flatten_experts_for_table(experts: list[dict]) -> pd.DataFrame:
    """One row per (expert x resource)."""
    rows = []
    for ex in experts:
        if ex["resources"]:
            for res in ex["resources"]:
                rows.append(
                    {
                        "author": ex["author"],
                        "website": ex["website"],
                        "headshot": ex["headshot"],
                        "chips": " | ".join(ex["chips"]),
                        "reason": ex["reason"],
                        "resource_label": res.get("label"),
                        "resource_url": res.get("url"),
                    }
                )
        else:
            rows.append(
                {
                    "author": ex["author"],
                    "website": ex["website"],
                    "headshot": ex["headshot"],
                    "chips": " | ".join(ex["chips"]),
                    "reason": ex["reason"],
                    "resource_label": "",
                    "resource_url": "",
                }
            )
    return pd.DataFrame(rows)

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
st.title(APP_TITLE)

tabs = st.tabs(["🔎 Scrape Experts Grid", "📝 Check Author-Submitted Content"])

# --- TAB 1: Scrape & evaluate resources ---
with tabs[0]:
    st.subheader("Scrape the experts grid from mlforseo.com/experts and evaluate authorship of each resource.")
    url = st.text_input("Experts page URL", value="https://www.mlforseo.com/experts/")
    colA, colB = st.columns([1, 1])
    with colA:
        do_eval = st.checkbox("Evaluate authorship of each resource", value=True, help="Fetch each resource URL and try to confirm the expert is the author.")
    with colB:
        show_headshots = st.checkbox("Show headshots", value=False)

    if st.button("Scrape Experts", type="primary"):
        experts, diags = scrape_experts(url)

        total = len(experts)
        if total:
            st.success(f"Found {total} expert(s).")
        else:
            st.info("Found 0 experts.")

        # Table
        df = flatten_experts_for_table(experts)

        # Optional author evaluation per resource
        eval_rows = []
        if do_eval and not df.empty:
            st.caption("Evaluating authorship…")
            for _, row in df.iterrows():
                res_url = row["resource_url"]
                expected = row["author"]
                if res_url:
                    ev = evaluate_authorship(res_url, expected)
                else:
                    ev = {
                        "url": "",
                        "http_status": None,
                        "expected_author": expected,
                        "authors_detected": [],
                        "methods": [],
                        "author_match": None,
                        "matched_author": "",
                        "confidence": 0.0,
                        "evidence_method": "",
                        "error": None,
                    }
                ev_row = {
                    "resource_url": ev["url"],
                    "expected_author": ev["expected_author"],
                    "author_match": ev["author_match"],
                    "matched_author": ev["matched_author"],
                    "confidence": ev["confidence"],
                    "authors_detected": " | ".join(ev["authors_detected"] or []),
                    "evidence_method": ev["evidence_method"],
                    "http_status": ev["http_status"],
                    "error": ev.get("error"),
                }
                eval_rows.append(ev_row)

            eval_df = pd.DataFrame(eval_rows)
            df = df.merge(eval_df, on="resource_url", how="left")

        # Visuals
        if show_headshots and not df.empty:
            # Streamlit tables don't render images inline; just show URL as text.
            st.caption("Headshot URLs are shown as text for export; toggle off if noisy.")

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Downloads
        st.download_button(
            "Download JSON",
            data=json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            file_name="experts_grid.json",
            mime="application/json",
        )
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="experts_grid.csv",
            mime="text/csv",
        )

        with st.expander("Diagnostics (first 2 experts)"):
            st.json({**diags, "sample": (experts[:2] if experts else [])})

# --- TAB 2: Author submitted content checker ---
with tabs[1]:
    st.subheader("Verify that submitted content URLs are authored by the given person.")
    c1, c2 = st.columns([1, 2])
    with c1:
        author_name = st.text_input("Author Name", value="", placeholder="e.g., Aimee Jurenka")
    with c2:
        urls_csv = st.text_area(
            "Content URLs (comma-separated)",
            value="",
            placeholder="https://example.com/post-1, https://example.com/post-2",
            height=80,
        )
    run = st.button("Run Checks", type="primary")

    if run:
        urls = [u.strip() for u in urls_csv.split(",") if u.strip()]
        rows = []
        for u in urls:
            res = evaluate_authorship(u, author_name)
            rows.append(
                {
                    "url": res["url"],
                    "expected_author": res["expected_author"],
                    "author_match": res["author_match"],
                    "matched_author": res["matched_author"],
                    "confidence": res["confidence"],
                    "authors_detected": " | ".join(res["authors_detected"] or []),
                    "evidence_method": res["evidence_method"],
                    "http_status": res["http_status"],
                    "error": res.get("error"),
                }
            )
        out_df = pd.DataFrame(rows)
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Results (CSV)",
            data=out_df.to_csv(index=False),
            file_name="author_checks.csv",
            mime="text/csv",
        )
        with st.expander("Raw JSON"):
            st.json(rows)
