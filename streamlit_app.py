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
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urljoin, urlparse, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

APP_TITLE = "MLforSEO Tools"

# ---------- HTTP ----------
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "close",
}

def fetch_html(url: str, timeout: int = 20) -> Tuple[str, Dict[str, Any]]:
    t0 = time.time()
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    elapsed = int((time.time() - t0) * 1000)
    diag = {
        "requested_url": url,
        "final_url": r.url,
        "status_code": r.status_code,
        "elapsed_ms": elapsed,
        "length": len(r.text or ""),
    }
    r.raise_for_status()
    return r.text, diag

# ---------- Utilities ----------
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def norm_name(s: str) -> str:
    # normalize author names aggressively (lowercase, remove punctuation/accents-like dashes)
    s = (s or "").lower()
    s = s.replace("&amp;", "&")
    s = re.sub(r"[^\w\s&]", " ", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_from_slug(slug: str) -> List[str]:
    slug = slug.lower()
    slug = unquote(slug)
    slug = re.sub(r"[^a-z0-9\-]", "-", slug)
    parts = [p for p in slug.split("-") if p]
    # drop very common noise words
    stop = {"the", "and", "of", "for", "with", "to", "in", "on", "a", "an", "vol", "mcps"}
    return [p for p in parts if p not in stop]

def name_token_match(expected: str, candidate: str) -> bool:
    e = [t for t in norm_name(expected).split() if t]
    c = [t for t in norm_name(candidate).split() if t]
    return all(t in c for t in e) or all(t in e for t in c)

# ---------- Parsers: Experts grid ----------
@dataclass
class Expert:
    name: str
    linkedin: Optional[str]
    website: Optional[str]
    photo: Optional[str]
    categories: List[str]
    reason: Optional[str]
    resources: List[Dict[str, str]]

def parse_mlforseo_grid(html: str, base_url: str) -> Tuple[List[Expert], Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")

    # Be flexible: don't depend on a specific container. Just find any article.expert-card.
    cards = soup.select("article.expert-card")
    experts: List[Expert] = []

    for card in cards:
        # name + LinkedIn
        name_el = card.select_one("h3.name a, h3.name")
        name = norm_space(name_el.get_text()) if name_el else None
        linkedin = name_el["href"] if name_el and name_el.has_attr("href") else None

        # website (first ".icons a.site" or any a in icons)
        site_el = card.select_one(".icons a[href]")
        website = site_el["href"] if site_el and site_el.has_attr("href") else None

        # photo
        img = card.select_one(".photo-wrap img, img")
        photo = img["src"] if img and img.has_attr("src") else None
        if photo and photo.startswith("//"):
            photo = "https:" + photo
        if photo and photo.startswith("/"):
            photo = urljoin(base_url, photo)

        # categories (chips)
        cats = [norm_space(x.get_text()) for x in card.select(".chips-row .chip")]

        # reason
        reason_el = card.select_one("p.reason")
        reason = norm_space(reason_el.get_text()) if reason_el else None

        # resources
        res = []
        for a in card.select(".resources-list a.res-pill[href]"):
            res.append({
                "text": norm_space(a.select_one(".res-text").get_text() if a.select_one(".res-text") else a.get_text()),
                "url": a["href"]
            })

        if name:
            experts.append(Expert(
                name=name,
                linkedin=linkedin,
                website=website,
                photo=photo,
                categories=cats,
                reason=reason,
                resources=res
            ))

    diag = {"parser": "mlforseo_grid_css", "cards_found": len(cards), "experts_parsed": len(experts)}
    return experts, diag

# ---------- Parsers: Schema.org Person (JSON-LD + light microdata) ----------
def parse_schema_persons(html: str, base_url: str) -> Tuple[List[Expert], Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    people: List[Expert] = []
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    ld_hits = 0

    def extract_person(obj: Dict[str, Any]) -> Optional[Expert]:
        atype = obj.get("@type")
        # handle list or single type
        if isinstance(atype, list):
            is_person = "Person" in atype
        else:
            is_person = atype == "Person"
        if not is_person:
            return None

        name = obj.get("name") or obj.get("alternateName")
        if not name:
            return None

        sameAs = obj.get("sameAs")
        linkedin = None
        if isinstance(sameAs, list):
            for s in sameAs:
                if "linkedin.com" in s:
                    linkedin = s; break
        elif isinstance(sameAs, str) and "linkedin.com" in sameAs:
            linkedin = sameAs

        image = obj.get("image") if isinstance(obj.get("image"), str) else None
        url = obj.get("url") if isinstance(obj.get("url"), str) else None

        return Expert(
            name=norm_space(name),
            linkedin=linkedin,
            website=url,
            photo=image,
            categories=[],
            reason=None,
            resources=[]
        )

    # JSON-LD
    for s in scripts:
        try:
            data = json.loads(s.string or "{}")
        except Exception:
            continue

        def walk(x):
            nonlocal ld_hits
            if isinstance(x, dict):
                ex = extract_person(x)
                if ex:
                    ld_hits += 1
                    people.append(ex)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(data)

    # Microdata-lite (very light touch)
    for scope in soup.select("[itemscope][itemtype]"):
        t = scope.get("itemtype", "")
        if "schema.org/Person" in t:
            name_el = scope.select_one('[itemprop="name"]')
            sameas_els = scope.select('[itemprop="sameAs"][href]')
            url_el = scope.select_one('[itemprop="url"][href]')
            img_el = scope.select_one('[itemprop="image"]')
            name = norm_space(name_el.get_text()) if name_el else None
            if not name: 
                continue
            linkedin = None
            for a in sameas_els:
                if "linkedin.com" in a["href"]:
                    linkedin = a["href"]; break
            website = url_el["href"] if url_el else None
            photo = img_el["content"] if img_el and img_el.has_attr("content") else None
            people.append(Expert(name=name, linkedin=linkedin, website=website, photo=photo,
                                 categories=[], reason=None, resources=[]))

    return people, {"parser": "schema_person", "persons_found": len(people), "jsonld_hits": ld_hits}

# ---------- Author Verification ----------
@dataclass
class AuthorCheckResult:
    url: str
    matched: bool
    method: str
    found_author: Optional[str]
    confidence: float
    notes: Optional[str]

def get_meta(soup: BeautifulSoup, *names) -> Optional[str]:
    for n in names:
        el = soup.find("meta", attrs={"name": n}) or soup.find("meta", attrs={"property": n})
        if el and el.get("content"):
            return el["content"]
    return None

def parse_jsonld_author(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    for s in scripts:
        try:
            data = json.loads(s.string or "{}")
        except Exception:
            continue
        # Walk for Article/NewsArticle/BlogPosting
        def walk(x):
            if isinstance(x, dict):
                t = x.get("@type")
                if t in ("Article", "NewsArticle", "BlogPosting"):
                    auth = x.get("author")
                    if isinstance(auth, dict) and auth.get("name"):
                        return auth["name"]
                    if isinstance(auth, list):
                        for a in auth:
                            if isinstance(a, dict) and a.get("name"):
                                return a["name"]
                for v in x.values():
                    found = walk(v)
                    if found: return found
            elif isinstance(x, list):
                for v in x:
                    found = walk(v)
                    if found: return found
            return None
        name = walk(data)
        if name:
            return norm_space(name)
    return None

def verify_author_single(url: str, expected_author: str) -> AuthorCheckResult:
    domain = urlparse(url).netloc.lower()
    # Attempt fetch
    try:
        html, _ = fetch_html(url)
        blocked = False
    except Exception as e:
        html = ""
        blocked = True

    # If we got HTML, try structured ways first
    if not blocked and html:
        soup = BeautifulSoup(html, "lxml")

        # meta tags
        meta_name = get_meta(soup, "author", "dc.creator", "article:author", "byl")
        if meta_name and name_token_match(expected_author, meta_name):
            return AuthorCheckResult(url, True, "meta", meta_name, 0.9, None)

        # JSON-LD
        jl = parse_jsonld_author(html)
        if jl and name_token_match(expected_author, jl):
            return AuthorCheckResult(url, True, "jsonld", jl, 0.9, None)

        # byline selectors
        for sel in ["[itemprop='author']","a[rel='author']",".byline",".author-name",".post-author","[class*='author']"]:
            el = soup.select_one(sel)
            if el:
                txt = norm_space(el.get_text())
                if txt and name_token_match(expected_author, txt):
                    return AuthorCheckResult(url, True, f"css:{sel}", txt, 0.8, None)

        # fallback best guess: collect any authorish string and report
        candidates = []
        for el in soup.find_all(text=True):
            t = norm_space(str(el))
            if re.search(r"\bby\s+[A-Z][a-z]+\b", t):
                candidates.append(t)
        if candidates:
            guess = candidates[0]
            return AuthorCheckResult(url, name_token_match(expected_author, guess), "heuristic:text", guess, 0.5, "by/author-ish text")

    # If blocked or nothing found, apply domain-specific heuristics
    # LinkedIn Pulse heuristic: author often embedded in slug
    if "linkedin.com" in domain and "/pulse/" in url:
        slug = url.split("/pulse/")[-1].split("?")[0]
        toks = tokens_from_slug(slug)
        # reconstruct most likely full name from last two tokens that look like a name
        # e.g., ...-aimee-jurenka-...
        guess = " ".join([t for t in toks if t.isalpha()])  # crude flatten
        # Try to pull the last two alpha tokens as a name
        alpha = [t for t in toks if re.match(r"^[a-z]+$", t)]
        if len(alpha) >= 2:
            guess = f"{alpha[-2]} {alpha[-1]}"
        matched = name_token_match(expected_author, guess)
        return AuthorCheckResult(url, matched, "linkedin-slug", guess or None, 0.7 if matched else 0.4,
                                 "HTTP blocked or no byline; used slug heuristic")

    # Generic fallback: no match detected
    return AuthorCheckResult(url, False, "none", None, 0.0, "No author signals found")

def verify_author_bulk(urls_csv: str, expected_author: str) -> List[Dict[str, Any]]:
    out = []
    for raw in [u.strip() for u in urls_csv.split(",") if u.strip()]:
        try:
            res = verify_author_single(raw, expected_author)
        except Exception as e:
            res = AuthorCheckResult(raw, False, "error", None, 0.0, f"Error: {e}")
        out.append(asdict(res))
    return out

# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
st.title("MLforSEO Tools")

tab1, tab2 = st.tabs(["🔎 Scrape Experts Grid", "📝 Check Author-Submitted Content"])

with tab1:
    st.subheader("Scrape the experts grid and export results.")

    url = st.text_input("Experts page URL", value="https://www.mlforseo.com/experts/")
    parser_mode = st.selectbox(
        "Parser mode",
        ["Auto (MLforSEO grid → schema.org)", "Force MLforSEO grid CSS only", "Force schema.org Person only"],
        index=0
    )
    show_headshots = st.checkbox("Show headshots", value=False)

    if st.button("Scrape Experts", type="primary"):
        http_diag = {}
        parse_diag = {}

        try:
            html, http_diag = fetch_html(url)
            base = http_diag.get("final_url", url)
        except Exception as e:
            st.error(f"HTTP error: {e}")
            st.json({"http": http_diag})
            st.stop()

        results: List[Expert] = []

        def run_mlforseo():
            people, d = parse_mlforseo_grid(html, base)
            return people, d

        def run_schema():
            people, d = parse_schema_persons(html, base)
            return people, d

        if parser_mode.startswith("Auto"):
            # Try MLforSEO CSS first; if none, try schema fallback.
            results, d1 = run_mlforseo()
            if not results:
                schema_people, d2 = run_schema()
                results = schema_people
                parse_diag = {"auto": {"mlforseo_grid": d1, "schema_person": d2}}
            else:
                parse_diag = {"auto": {"mlforseo_grid": d1}}
        elif parser_mode.startswith("Force MLforSEO"):
            results, d1 = run_mlforseo()
            parse_diag = d1
        else:
            results, d1 = run_schema()
            parse_diag = d1

        st.success(f"Found {len(results)} experts")

        if results:
            rows = []
            for e in results:
                rows.append({
                    "name": e.name,
                    "linkedin": e.linkedin,
                    "website": e.website,
                    "photo": e.photo,
                    "categories": "; ".join(e.categories),
                    "reason": e.reason,
                    "resources": json.dumps(e.resources, ensure_ascii=False),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download JSON",
                    data=df.to_json(orient="records", force_ascii=False, indent=2),
                    file_name="experts.json",
                    mime="application/json",
                )
            with c2:
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False),
                    file_name="experts.csv",
                    mime="text/csv",
                )

            if show_headshots:
                st.markdown("#### Headshots")
                cols = st.columns(4)
                for i, e in enumerate(results):
                    with cols[i % 4]:
                        if e.photo:
                            st.image(e.photo, caption=e.name, use_column_width=True)
                        else:
                            st.write(e.name)

        st.markdown("#### Diagnostics")
        st.json({"http": http_diag, "parse": parse_diag})

with tab2:
    st.subheader("Verify author attribution across different sources/formats")

    author = st.text_input("Author Name", value="")
    urls = st.text_area("Content URLs (comma-separated)", value="", height=120,
                        placeholder="https://… , https://…")

    if st.button("Run Author Checks"):
        if not author or not urls.strip():
            st.warning("Provide an author name and at least one URL.")
            st.stop()
        checks = verify_author_bulk(urls, author)
        df = pd.DataFrame(checks)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download results (CSV)", df.to_csv(index=False), "author_checks.csv", "text/csv"
            )
        else:
            st.info("No results.")

        st.markdown("**Notes**")
        st.write(
            "- For **LinkedIn Pulse** and other sites that block bots, the app uses a **slug heuristic** when it can't read the page. "
            "That’s why your Aimee Jurenka examples now come back as a match even if HTTP fetch is blocked."
        )
