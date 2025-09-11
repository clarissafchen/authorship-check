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
import streamlit as st
from bs4 import BeautifulSoup

# ---- Playwright (JS rendering) ----
USE_JS_DEFAULT = True
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

def fetch_html(url: str, use_js: bool = True, timeout_ms: int = 15000):
    """
    Fetch HTML. Use Playwright if available & requested, else fallback to requests.
    Returns dict: {html, status, final_url, engine, error}
    """
    if use_js:
        try:
            from playwright.sync_api import sync_playwright
            t0 = time.time()
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
                ctx = browser.new_context(user_agent=UA, java_script_enabled=True)
                page = ctx.new_page()
                resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                html = page.content()
                out = {
                    "html": html,
                    "status": resp.status if resp else None,
                    "final_url": page.url,
                    "engine": "playwright",
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "error": None,
                }
                ctx.close()
                browser.close()
                return out
        except Exception as e:
            # fallthrough to requests
            err = f"Playwright error: {e}"

    # fallback: requests
    import requests
    t0 = time.time()
    try:
        r = requests.get(
            url,
            headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
            timeout=20,
            allow_redirects=True,
        )
        return {
            "html": r.text if r.status_code == 200 else "",
            "status": r.status_code,
            "final_url": r.url,
            "engine": "requests",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "error": None,
        }
    except Exception as e:
        return {"html": "", "status": None, "final_url": url, "engine": "requests", "elapsed_ms": 0, "error": str(e)}


# ---- Name/author utilities ----
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if unicodedata.category(c) != "Mn")

def _norm(s: str) -> str:
    s = _strip_accents(s or "")
    s = re.sub(r"[\u2010-\u2015\-–—]", " ", s)
    s = re.sub(r"[^a-zA-Z\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _tokens(name: str) -> list[str]:
    return [t for t in _norm(name).split() if len(t) > 1]

def _token_sim(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)

def names_match(expected: str, candidate: str) -> tuple[bool, float]:
    te, tc = _tokens(expected), _tokens(candidate)
    if not te or not tc:
        return (False, 0.0)
    if set(te).issubset(set(tc)) or set(tc).issubset(set(te)):
        return (True, 0.95 if te[-1] == tc[-1] else 0.85)
    sim = _token_sim(te, tc)
    last_ok = te[-1] == tc[-1]
    first_initial = te[0][0] == tc[0][0]
    if sim >= 0.67 or (last_ok and (first_initial or te[0] == tc[0])):
        return (True, max(sim, 0.75 if last_ok else 0.7))
    return (False, sim)


# ---- Byline extraction ----
BYLINE_SELECTORS = [
    '[itemprop="author"]', '[rel="author"]',
    '.byline', '.by-line', '.entry-author', '.post-author',
    '.article-author', '.article__author', '.author-name', '.author',
    '.meta-author', '[data-testid*="author"]', '[data-test*="author"]',
    '[class*="author"]', '[class*="byline"]',
    '[class*="creator"]', '[data-testid*="creator"]', '[data-test*="creator"]',
]

def _walk_json_for_names(obj) -> list[str]:
    out = []
    keys = ("author", "creator", "accountablePerson", "uploader", "byline")
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.lower()
            if any(x in lk for x in keys):
                if isinstance(v, str) and 1 < len(v) < 120:
                    out.append(v)
                elif isinstance(v, dict):
                    nm = v.get("name") or v.get("alternateName") or v.get("title") or ""
                    if 1 < len(nm) < 120:
                        out.append(nm)
                elif isinstance(v, list):
                    for i in v:
                        if isinstance(i, str) and 1 < len(i) < 120:
                            out.append(i)
                        elif isinstance(i, dict):
                            nm = i.get("name") or i.get("alternateName") or i.get("title") or ""
                            if 1 < len(nm) < 120:
                                out.append(nm)
            out.extend(_walk_json_for_names(v))
    elif isinstance(obj, list):
        for i in obj:
            out.extend(_walk_json_for_names(i))
    return out

def extract_authors_jsonld(soup: BeautifulSoup) -> list[str]:
    out = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(s.string or s.text)
            out.extend(_walk_json_for_names(data))
        except Exception:
            pass
    return out

def extract_authors_meta(soup: BeautifulSoup) -> list[str]:
    out = []
    for sel_attr, val in (("name", "author"), ("property", "article:author")):
        for m in soup.select(f'meta[{sel_attr}="{val}"]'):
            c = (m.get("content") or "").strip()
            if c and len(c) < 120:
                out.append(c)
    return out

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
    path = urlparse(url).path.lower()
    # LinkedIn Pulse: /pulse/<title>-firstname-lastname-<hash>
    m = re.search(r"/pulse/[^/]*-([a-z]+)-([a-z]+)-[a-z0-9]+/?$", path)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    # Generic last two alpha tokens in slug
    slug = path.strip("/").split("/")[-1]
    slug = re.sub(r"\.(html|htm)$", "", slug)
    toks = [t for t in slug.split("-") if t.isalpha()]
    if len(toks) >= 2:
        return f"{toks[-2]} {toks[-1]}"
    return ""

def extract_authors_site_specific(url: str, soup: BeautifulSoup) -> list[str]:
    host = urlparse(url).netloc.lower()
    out = []
    # Loom: Next.js dump often contains creator info
    if "loom.com" in host:
        tag = soup.find("script", id="__NEXT_DATA__")
        if tag and (tag.string or tag.text):
            try:
                data = json.loads(tag.string or tag.text)
                out.extend(_walk_json_for_names(data))
            except Exception:
                pass
    # Women in Tech SEO: explicit "Author:" label
    if "womenintechseo.com" in host:
        for el in soup.find_all(string=re.compile(r"Author:", re.I)):
            txt = el.parent.get_text(" ", strip=True)
            nm = re.sub(r".*Author:\s*", "", txt, flags=re.I).strip()
            if 1 < len(nm) < 120:
                out.append(nm)
    return out

def detect_authors(url: str, use_js: bool = False) -> dict:
    resp = fetch_html(url, use_js=use_js)
    html = resp.get("html", "")
    soup = BeautifulSoup(html or "", "lxml") if html else None

    authors, methods = [], []
    if soup:
        a = extract_authors_jsonld(soup)
        if a: authors += a; methods += ["jsonld"] * len(a)

        a = extract_authors_meta(soup)
        if a: authors += a; methods += ["meta"] * len(a)

        a = extract_authors_bylines(soup)
        if a: authors += a; methods += ["byline"] * len(a)

        a = extract_authors_site_specific(url, soup)
        if a: authors += a; methods += ["site"] * len(a)

    slug_guess = guess_author_from_url(url)
    if slug_guess:
        authors.insert(0, slug_guess)
        methods.insert(0, "slug")

    # Dedup (order-preserving)
    seen, uniq, uniq_m = set(), [], []
    for nm, m in zip(authors, methods):
        key = _norm(nm)
        if key and key not in seen:
            seen.add(key); uniq.append(nm.strip()); uniq_m.append(m)

    return {
        "url": url,
        "http_status": resp.get("status"),
        "engine": resp.get("engine"),
        "authors": uniq,
        "methods": uniq_m,
        "error": resp.get("error"),
        "html_used": bool(html),
    }


# ---- Profile/About page detection ----
PROFILE_URL_PATTERNS = re.compile(
    r"(?:/author/|/authors?/|/about|/profile|/users?/|/people/|/team/|/contributors?/|/staff/|/in/)",
    re.I,
)
PROFILE_DOMAINS = (
    "linkedin.com/in/",
    "x.com/",
    "twitter.com/",
    "github.com/",
    "substack.com/@",
    "medium.com/@",
)

def is_profile_page(url: str, soup: BeautifulSoup | None) -> tuple[bool, str]:
    path = urlparse(url).path.lower()
    full = url.lower()
    if PROFILE_URL_PATTERNS.search(path):
        return True, "url_pattern"
    if any(d in full for d in PROFILE_DOMAINS):
        return True, "known_profile_domain"
    if soup:
        # og:type = profile
        for m in soup.select('meta[property="og:type"]'):
            if (m.get("content") or "").strip().lower() == "profile":
                return True, "og_profile"
        # JSON-LD type Person/ProfilePage/AboutPage without Article/BlogPosting
        try:
            for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
                data = json.loads(s.string or s.text)
                if isinstance(data, dict):
                    types = {data.get("@type", "")}
                elif isinstance(data, list):
                    types = set()
                    for d in data:
                        if isinstance(d, dict) and d.get("@type"):
                            t = d["@type"]
                            if isinstance(t, list): types.update(t)
                            else: types.add(t)
                else:
                    types = set()
                types = {str(t).lower() for t in types}
                if types and not ({"article","blogposting","newsarticle","videoobject"} & types):
                    if {"person","profilepage","aboutpage","contactpage"} & types:
                        return True, "jsonld_profile"
        except Exception:
            pass
    return False, ""


# ---- Experts grid parsing ----
def parse_experts_grid(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    cards = soup.select("div.experts-grid article.expert-card")
    experts = []
    for card in cards:
        author_node = (card.select_one("h3.name a") or card.select_one("h3.name"))
        author = (author_node.get_text(strip=True) if author_node else "").strip()

        site = card.select_one(".icons a.icon-link.site")
        website = site.get("href").strip() if site and site.has_attr("href") else ""

        img = card.select_one(".photo-wrap img")
        headshot = img.get("src").strip() if img and img.has_attr("src") else ""

        chips = [c.get_text(strip=True) for c in card.select(".chips-row .chip")]

        reason = card.select_one("p.reason")
        reason = reason.get_text(" ", strip=True) if reason else ""

        resources = []
        for a in card.select(".resources-list a.res-pill"):
            href = a.get("href")
            if not href: continue
            label = a.select_one(".res-text")
            label_txt = label.get_text(strip=True) if label else urlparse(href).netloc
            resources.append({"url": href.strip(), "label": label_txt})

        experts.append(
            {
                "author": author,        # <- renamed
                "website": website,
                "headshot": headshot,
                "chips": chips,
                "reason": reason,
                "resources": resources,
            }
        )
    return experts

def flatten_experts(experts: list[dict]) -> pd.DataFrame:
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


# ---- Authorship evaluation ----
def evaluate_authorship(url: str, expected_author: str, use_js: bool = False) -> dict:
    resp = fetch_html(url, use_js=use_js)
    html = resp.get("html", "")
    soup = BeautifulSoup(html or "", "lxml") if html else None

    # profile/about filter
    prof, reason = is_profile_page(resp.get("final_url") or url, soup)
    if prof:
        return {
            "url": resp.get("final_url") or url,
            "expected_author": expected_author,
            "author_match": False,
            "matched_author": "",
            "confidence": 0.0,
            "authors_detected": [],
            "evidence_method": "",
            "http_status": resp.get("status"),
            "engine": resp.get("engine"),
            "error": None,
            "is_profile_page": True,
            "profile_reason": reason,
        }

    det = detect_authors(resp.get("final_url") or url, use_js=False)  # bylines usually OK w/o JS
    best = {"cand": "", "conf": 0.0, "ok": False, "method": ""}
    for cand, m in zip(det["authors"], det.get("methods", [])):
        ok, score = names_match(expected_author, cand)
        if ok and score > best["conf"]:
            best = {"cand": cand, "conf": score, "ok": True, "method": m}

    if not best["ok"] and det["authors"]:
        # pick highest similarity candidate
        sims = [(_token_sim(_tokens(expected_author), _tokens(c)), c, m) for c, m in zip(det["authors"], det.get("methods", []))]
        sims.sort(reverse=True, key=lambda x: x[0])
        _, c, m = sims[0]
        best = {"cand": c, "conf": _, "ok": False, "method": m}

    return {
        "url": resp.get("final_url") or url,
        "expected_author": expected_author,
        "author_match": best["ok"],
        "matched_author": best["cand"],
        "confidence": round(float(best["conf"]), 3),
        "authors_detected": det.get("authors"),
        "evidence_method": best["method"],
        "http_status": resp.get("status"),
        "engine": resp.get("engine"),
        "error": det.get("error") or resp.get("error"),
        "is_profile_page": False,
        "profile_reason": "",
    }


# -------------------- UI --------------------
st.set_page_config(page_title="MLforSEO Tools", page_icon="🧭", layout="wide")
st.title("MLforSEO Tools")
tab1, tab2 = st.tabs(["🔎 Scrape Experts Grid", "📝 Check Author-Submitted Content"])

# Tab 1
with tab1:
    st.subheader("Scrape the experts grid and (optionally) validate authorship of each resource.")
    colA, colB, colC = st.columns([2,1,1])
    with colA:
        experts_url = st.text_input("Experts page URL", value="https://www.mlforseo.com/experts/")
    with colB:
        use_js = st.checkbox("Use headless browser (Playwright)", value=USE_JS_DEFAULT)
    with colC:
        do_eval = st.checkbox("Evaluate authorship", value=True)

    if st.button("Scrape Experts", type="primary"):
        # fetch grid
        grid_resp = fetch_html(experts_url, use_js=use_js)
        diags = {"http": {k: grid_resp.get(k) for k in ("final_url","status","engine","elapsed_ms")}}
        experts = parse_experts_grid(grid_resp.get("html","")) if grid_resp.get("html") else []
        diags["parse"] = {"cards_found": len(experts), "parser": "mlforseo_grid_css"}

        st.success(f"Found {len(experts)} expert(s).") if experts else st.info("Found 0 experts.")
        df = flatten_experts(experts)

        # evaluate authorship per resource (skip blank URLs)
        if do_eval and not df.empty:
            eval_rows = []
            for _, row in df.iterrows():
                url = row["resource_url"]
                if not url:
                    eval_rows.append(
                        {"resource_url":"", "author_match":None, "matched_author":"", "confidence":0.0,
                         "authors_detected":"", "evidence_method":"", "http_status":None, "engine":"", "is_profile_page":None, "profile_reason":""}
                    )
                    continue
                res = evaluate_authorship(url, row["author"], use_js=False)
                eval_rows.append(
                    {
                        "resource_url": res["url"],
                        "author_match": res["author_match"],
                        "matched_author": res["matched_author"],
                        "confidence": res["confidence"],
                        "authors_detected": " | ".join(res.get("authors_detected") or []),
                        "evidence_method": res["evidence_method"],
                        "http_status": res["http_status"],
                        "engine": res["engine"],
                        "is_profile_page": res["is_profile_page"],
                        "profile_reason": res["profile_reason"],
                    }
                )
            eval_df = pd.DataFrame(eval_rows)
            df = df.merge(eval_df, on="resource_url", how="left")

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download JSON",
            json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            file_name="experts_grid.json",
            mime="application/json",
        )
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="experts_grid.csv",
            mime="text/csv",
        )

        with st.expander("Diagnostics"):
            st.json(diags)

# Tab 2
with tab2:
    st.subheader("Verify that submitted URLs are actually authored by the person (and not just a profile/about page).")
    c1, c2 = st.columns([1,2])
    with c1:
        author_name = st.text_input("Author Name", value="")
    with c2:
        urls_csv = st.text_area("Content URLs (comma-separated)", value="", height=100)

    if st.button("Run Checks", type="primary"):
        urls = [u.strip() for u in urls_csv.split(",") if u.strip()]
        rows = []
        for u in urls:
            res = evaluate_authorship(u, author_name, use_js=False)
            rows.append(
                {
                    "url": res["url"],
                    "expected_author": res["expected_author"],
                    "author_match": res["author_match"],
                    "matched_author": res["matched_author"],
                    "confidence": res["confidence"],
                    "authors_detected": " | ".join(res.get("authors_detected") or []),
                    "evidence_method": res["evidence_method"],
                    "is_profile_page": res["is_profile_page"],
                    "profile_reason": res["profile_reason"],
                    "http_status": res["http_status"],
                    "engine": res["engine"],
                    "error": res.get("error"),
                }
            )
        out_df = pd.DataFrame(rows)
        st.dataframe(out_df, use_container_width=True, hide_index=True)
        st.download_button("Download Results (CSV)", out_df.to_csv(index=False), "author_checks.csv", "text/csv")
        with st.expander("Raw JSON"):
            st.json(rows)
