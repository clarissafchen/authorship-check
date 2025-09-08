# Streamlit app:
#   - Tab 1: Scrape https://www.mlforseo.com/experts/ grid (requests + BeautifulSoup)
#   - Tab 2: Verify authorship of submitted content URLs against a provided author name
#
# Notes:
# - Author check gathers evidence from JSON-LD, meta tags, visible bylines, domain adapters
# - LinkedIn Pulse often blocks bots; we add slug heuristics + JSON-LD regex fallback
# - Matching uses normalized tokens + fuzzy ratio (difflib) to classify MATCH/POSSIBLE/NO_MATCH

import re
import time
import json
import html
import unicodedata
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

st.set_page_config(page_title="MLforSEO Tools", layout="wide")

# -------------------- Shared HTTP utils --------------------

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

def _normalize_url(href: str, base: str) -> str:
    if not href:
        return ""
    href = href.strip()
    if href.startswith(("http://", "https://")):
        return href
    if href.startswith("//"):
        return "https:" + href
    # bare domain/path like linkedin.com/..., github.com/...
    if re.match(r"^[\w.-]+\.[a-z]{2,}(/|$)", href, flags=re.I):
        return "https://" + href
    return urljoin(base, href)

def _strip_tracking(u: str) -> str:
    """Remove common tracking query params but preserve core path."""
    try:
        parsed = urlparse(u)
        qs = parse_qs(parsed.query)
        # Keep nothing unless specifically useful; LinkedIn trackingId not needed.
        cleaned = parsed._replace(query="")
        return urlunparse(cleaned)
    except Exception:
        return u

def _fetch_html(url: str, retries: int = 3, timeout: int = 15, sleep_factor: float = 1.3) -> str:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r.text
            # common anti-bot/ratelimit statuses (LinkedIn returns 999 sometimes; treat like 429)
            if r.status_code in (403, 429, 503, 999):
                time.sleep(sleep_factor * (i + 1))
                continue
            r.raise_for_status()
        except Exception as e:
            last_err = e
            time.sleep(sleep_factor * (i + 1))
    raise RuntimeError(f"Failed to fetch HTML from {url}: {last_err}")

def _soup(html_text: str) -> BeautifulSoup:
    return BeautifulSoup(html_text, "lxml")

# -------------------- Tab 1: Experts Grid Scraper --------------------

def _parse_expert_card(card, base_url: str) -> dict:
    name_a = card.select_one("h3.name a")
    img = card.select_one(".photo-wrap img")
    site_a = card.select_one(".icons a.site")

    data_cats = html.unescape(card.get("data-cats") or "")
    categories = [c.strip() for c in data_cats.split("|") if c.strip()]

    chips = [
        (chip.get("data-chip") or chip.get_text(strip=True))
        for chip in card.select(".chips-row .chip")
    ]

    resources = []
    for a in card.select(".resources-list a.res-pill"):
        raw = a.get("href") or ""
        url_abs = _normalize_url(raw, base_url)
        domain_el = a.select_one(".res-text")
        domain = (domain_el.get_text(strip=True)
                  if domain_el else urlparse(url_abs).netloc.replace("www.", ""))
        favicon = ""
        fav_el = a.select_one("img.res-favicon")
        if fav_el and fav_el.has_attr("src"):
            favicon = _normalize_url(fav_el["src"], base_url)
        resources.append({"url": url_abs, "domain": domain, "favicon": favicon})

    return {
        "name": name_a.get_text(strip=True) if name_a else "",
        "linkedin": _normalize_url(name_a.get("href") if name_a else "", base_url),
        "website": _normalize_url(site_a.get("href") if site_a else "", base_url),
        "image": _normalize_url(img.get("src") if img else "", base_url),
        "image_alt": img.get("alt", "") if img else "",
        "categories": categories,
        "chips": chips,
        "reason": (card.select_one(".reason").get_text(strip=True)
                   if card.select_one(".reason") else ""),
        "resources": resources,
    }

@st.cache_data(show_spinner=False, ttl=600)
def scrape_mlforseo_experts(url: str = "https://www.mlforseo.com/experts/") -> list[dict]:
    html_text = _fetch_html(url)
    soup = _soup(html_text)
    grid = soup.select_one("#experts-grid")
    if not grid:
        raise ValueError("Could not find #experts-grid on the page. The markup may have changed.")
    cards = grid.select("article.expert-card")
    return [_parse_expert_card(card, url) for card in cards]

# -------------------- Tab 2: Author Verification --------------------

# --- Name normalization + matching ---

_STOP_TITLES = {"mr", "mrs", "ms", "dr", "prof", "sir"}

def _normalize_name(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[\u2010-\u2015\-‐-‒–—―]+", "-", s)  # hyphen-ish to hyphen
    s = re.sub(r"[^\w\s\-']", " ", s)  # drop punctuation except hyphen/apostrophe
    s = re.sub(r"\s+", " ", s).strip().lower()
    # drop common titles
    tokens = [t for t in s.split() if t not in _STOP_TITLES]
    return " ".join(tokens)

def _name_tokens(name: str) -> list[str]:
    return [t for t in _normalize_name(name).split() if t]

def _string_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_name(a), _normalize_name(b)).ratio()

def _token_coverage(author_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not author_tokens:
        return 0.0
    a = set(author_tokens)
    c = set(candidate_tokens)
    return len(a & c) / max(1, len(a))

# --- Evidence extraction ---

def _extract_jsonld_candidates(soup: BeautifulSoup) -> list[str]:
    found = []
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except Exception:
            # sometimes multiple JSON objects concatenated or comments present
            # try to locate author names via regex
            txt = script.get_text("", strip=True)
            for m in re.finditer(r'"author"\s*:\s*(\{.*?\}|\[.*?\])', txt, flags=re.I|re.S):
                frag = m.group(1)
                # naive name scrapes
                for m2 in re.finditer(r'"name"\s*:\s*"([^"]+)"', frag, flags=re.I):
                    found.append(m2.group(1).strip())
            continue

        def harvest(obj):
            if isinstance(obj, dict):
                # author can be dict or list
                if "author" in obj:
                    au = obj["author"]
                    if isinstance(au, dict):
                        nm = au.get("name")
                        if nm: found.append(str(nm).strip())
                    elif isinstance(au, list):
                        for x in au:
                            if isinstance(x, dict) and x.get("name"):
                                found.append(str(x["name"]).strip())
                # Some sites put creator
                if "creator" in obj:
                    cr = obj["creator"]
                    if isinstance(cr, dict) and cr.get("name"):
                        found.append(str(cr["name"]).strip())
                    elif isinstance(cr, list):
                        for x in cr:
                            if isinstance(x, dict) and x.get("name"):
                                found.append(str(x["name"]).strip())
                # recurse shallowly to catch nested Article/WebPage
                for v in obj.values():
                    harvest(v)
            elif isinstance(obj, list):
                for it in obj:
                    harvest(it)

        harvest(data)
    return list({x for x in found if x})

def _extract_meta_candidates(soup: BeautifulSoup) -> list[str]:
    meta_names = [
        ('name', 'author'),
        ('property', 'article:author'),  # can be URL on some sites
        ('name', 'parsely-author'),
        ('name', 'byl'),                 # NYT-style "byline"
        ('name', 'dc.creator'),
        ('name', 'sailthru.author'),
    ]
    names = []
    for attr, val in meta_names:
        for m in soup.find_all("meta", {attr: val}):
            content = (m.get("content") or "").strip()
            if content:
                names.append(content)
    # article:author might be a URL; extract trailing name hint
    cleaned = []
    for x in names:
        if re.match(r"https?://", x):
            # try to pull a last segment or slug
            p = urlparse(x)
            last = p.path.rstrip("/").split("/")[-1].replace("-", " ").strip()
            if last:
                cleaned.append(last)
            else:
                cleaned.append(x)
        else:
            cleaned.append(x)
    return list({c for c in cleaned if c})

def _extract_byline_candidates(soup: BeautifulSoup) -> list[str]:
    sel = [
        '[rel="author"]',
        '[itemprop="author"]',
        '.byline', '.by-line', '.post-author', '.author', '.article-author',
        '.c-article__author', '.meta-author', '.entry-author', '.posted-by'
    ]
    candidates = []
    for s in sel:
        for el in soup.select(s):
            t = el.get_text(" ", strip=True)
            if not t:
                continue
            # remove common prefixes
            t = re.sub(r"^\s*by\s+", "", t, flags=re.I)
            # trim pipes/labels
            t = re.sub(r"(?:author|posted by)\s*[:·|-]\s*", "", t, flags=re.I)
            # keep short-ish strings
            if 2 <= len(t) <= 120:
                candidates.append(t)
    # Title pattern e.g., "Article title — by Jane Doe"
    ttl = soup.title.get_text(strip=True) if soup.title else ""
    for m in re.finditer(r"\bby\s+([A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)+)", ttl):
        candidates.append(m.group(1))
    return list({c for c in candidates if c})

def _slug_name_hint(url: str) -> str:
    try:
        p = urlparse(url)
        # LinkedIn Pulse often: /pulse/slug-words-aimee-jurenka-XXXXXXXX
        # Medium/Substack/WordPress: slug sometimes contains the author, but not always.
        parts = [x for x in p.path.split("/") if x]
        if not parts:
            return ""
        slug = parts[-1]
        slug = slug.split("?")[0]
        hint = slug.replace("-", " ")
        hint = re.sub(r"\d+", " ", hint)
        hint = re.sub(r"\s+", " ", hint).strip()
        return hint
    except Exception:
        return ""

def _domain_adapter_candidates(url: str, soup: BeautifulSoup, html_text: str) -> list[str]:
    host = urlparse(url).netloc.lower().replace("www.", "")
    cands = []
    if "linkedin.com" in host and "/pulse/" in url:
        # 1) JSON-LD often present but sometimes blocked – already handled
        # 2) Regex scrape for author.name inside any JSON in the HTML
        for m in re.finditer(r'"author"\s*:\s*(\{.*?\}|\[.*?\])', html_text, flags=re.I|re.S):
            frag = m.group(1)
            for m2 in re.finditer(r'"name"\s*:\s*"([^"]+)"', frag, flags=re.I):
                cands.append(m2.group(1).strip())
        # 3) Slug hint (last resort)
        cands.append(_slug_name_hint(url))
    elif "medium.com" in host:
        # Medium typically has JSON-LD author + rel="author"
        pass
    elif host.endswith("substack.com"):
        # Substack exposes name via meta[name=author] and JSON-LD
        pass
    # WordPress often yields meta author + byline already covered
    return [c for c in cands if c]

def _collect_author_candidates(url: str, html_text: str) -> dict:
    soup = _soup(html_text)
    cands = []
    sources = []

    j = _extract_jsonld_candidates(soup)
    if j:
        cands.extend(j); sources.append("jsonld")

    m = _extract_meta_candidates(soup)
    if m:
        cands.extend(m); sources.append("meta")

    b = _extract_byline_candidates(soup)
    if b:
        cands.extend(b); sources.append("byline")

    d = _domain_adapter_candidates(url, soup, html_text)
    if d:
        cands.extend(d); sources.append("domain_adapter")

    # clean + unique
    cands = [c for c in {x.strip(): None for x in cands if x and x.strip()}.keys()]
    return {"candidates": cands, "sources": sources, "soup": soup}

# --- Core verification ---

def _classify_author_match(target_name: str, url: str, html_text: str) -> dict:
    target_tokens = _name_tokens(target_name)
    out = {
        "url": url,
        "canonical_url": None,
        "domain": urlparse(url).netloc.replace("www.", ""),
        "decision": "NO_MATCH",
        "score_ratio": 0.0,
        "token_coverage": 0.0,
        "found_candidates": [],
        "evidence_sources": [],
        "notes": "",
    }

    # gather candidates
    collected = _collect_author_candidates(url, html_text)
    cands = collected["candidates"]
    out["evidence_sources"] = list({*collected["sources"]})

    # canonical if present
    soup = collected["soup"]
    can = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    if can and can.get("href"):
        out["canonical_url"] = _strip_tracking(_normalize_url(can["href"], url))
    else:
        out["canonical_url"] = _strip_tracking(url)

    # If nothing found, add slug hint as evidence
    if not cands:
        sh = _slug_name_hint(url)
        if sh:
            cands.append(sh)
            out["evidence_sources"].append("slug_hint")

    # Evaluate candidates
    best_ratio = 0.0
    best_cov = 0.0
    found_clean = []

    for cand in cands:
        ratio = _string_ratio(target_name, cand)
        cov = _token_coverage(target_tokens, _name_tokens(cand))
        found_clean.append({"candidate": cand, "ratio": round(ratio, 3), "coverage": round(cov, 3)})
        best_ratio = max(best_ratio, ratio)
        best_cov = max(best_cov, cov)

    out["found_candidates"] = found_clean
    out["score_ratio"] = round(best_ratio, 3)
    out["token_coverage"] = round(best_cov, 3)

    # Decision thresholds (tuned to be forgiving on LinkedIn)
    # Full match: strong ratio or full token coverage
    if best_cov >= 1.0 or best_ratio >= 0.90:
        out["decision"] = "MATCH"
    # Possible: partial token coverage or decent ratio
    elif best_cov >= 0.5 or best_ratio >= 0.75:
        out["decision"] = "POSSIBLE_MATCH"
    else:
        out["decision"] = "NO_MATCH"

    # Add small note for LinkedIn Pulse when blocked
    host = out["domain"]
    if "linkedin.com" in host and not any(x["ratio"] >= 0.75 or x["coverage"] >= 0.5 for x in found_clean):
        out["notes"] = "LinkedIn may be blocking bot access; relying on slug heuristics. Consider manual check or HTML upload."

    return out

@st.cache_data(show_spinner=False, ttl=600)
def verify_author_for_urls(author_name: str, urls: list[str]) -> list[dict]:
    results = []
    for u in urls:
        u = _strip_tracking(u.strip())
        if not u:
            continue
        try:
            html_text = _fetch_html(u, retries=3, timeout=20)
            res = _classify_author_match(author_name, u, html_text)
        except Exception as e:
            res = {
                "url": u,
                "canonical_url": _strip_tracking(u),
                "domain": urlparse(u).netloc.replace("www.", ""),
                "decision": "ERROR",
                "score_ratio": 0.0,
                "token_coverage": 0.0,
                "found_candidates": [],
                "evidence_sources": [],
                "notes": f"{e}",
            }
        results.append(res)
        time.sleep(0.6)  # be polite
    return results

# -------------------- UI --------------------

st.title("MLforSEO Tools")

tab1, tab2 = st.tabs(["🔎 Scrape Experts Grid", "📝 Check Author-Submitted Content"])

with tab1:
    st.markdown("Scrape the experts grid from **mlforseo.com/experts** and export results.")
    with st.sidebar:
        st.markdown("### Scraper Settings")
        url_experts = st.text_input("Experts page URL", "https://www.mlforseo.com/experts/")
        show_images = st.checkbox("Show headshots", value=False, help="Disable for faster tables")

    btn_scrape = st.button("Scrape Experts", type="primary")
    if btn_scrape:
        try:
            with st.spinner("Scraping experts…"):
                rows = scrape_mlforseo_experts(url_experts)
            st.success(f"Found {len(rows)} experts")

            def pack_list(xs): 
                return "; ".join([x for x in xs if x]) if isinstance(xs, list) else ""
            def pack_domains(rs):
                return "; ".join(sorted({r.get("domain", "") for r in rs if r.get("domain")}))

            table_rows = []
            for r in rows:
                table_rows.append({
                    "name": r["name"],
                    "linkedin": r["linkedin"],
                    "website": r["website"],
                    "image": r["image"],
                    "image_alt": r["image_alt"],
                    "categories": pack_list(r["categories"]),
                    "chips": pack_list(r["chips"]),
                    "reason": r["reason"],
                    "resources_domains": pack_domains(r["resources"]),
                    "resources_json": json.dumps(r["resources"], ensure_ascii=False),
                })

            df = pd.DataFrame(table_rows)

            if show_images and not df.empty:
                st.markdown("#### Headshots")
                imgs = [i for i in df["image"].tolist() if i]
                if imgs:
                    st.image(imgs, width=96)
                else:
                    st.info("No images found to display.")

            st.markdown("#### Results")
            st.dataframe(df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download JSON",
                    data=json.dumps(rows, indent=2, ensure_ascii=False),
                    file_name="mlforseo-experts.json",
                    mime="application/json",
                )
            with c2:
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False),
                    file_name="mlforseo-experts.csv",
                    mime="text/csv",
                )

            with st.expander("Diagnostics (first 2 records)"):
                st.code(json.dumps(rows[:2], indent=2, ensure_ascii=False))

        except Exception as e:
            st.error(f"Scrape failed: {e}")
            st.caption("If the page becomes client-rendered, we can add a Playwright fallback.")

with tab2:
    st.markdown("Paste an **Author Name** and one or more **Content URLs** to verify authorship.")
    colA, colB = st.columns([2, 3])
    with colA:
        author_input = st.text_input("Author Name", placeholder="e.g., Aimee Jurenka")
        st.caption("We'll match across JSON-LD, meta tags, visible bylines, and URL hints.")
    with colB:
        urls_input = st.text_area(
            "Content URLs (comma or newline separated)",
            placeholder="https://...\nhttps://..."
        )

    # Optional knobs
    with st.expander("Advanced"):
        st.caption("These affect only classification thresholds.")
        ratio_full = st.slider("High fuzzy ratio threshold (MATCH)", 0.70, 0.99, 0.90, 0.01)
        cov_full = st.slider("Full token coverage threshold (MATCH)", 0.50, 1.00, 1.00, 0.05)
        ratio_possible = st.slider("Possible fuzzy ratio threshold", 0.50, 0.95, 0.75, 0.01)
        cov_possible = st.slider("Possible token coverage threshold", 0.25, 0.95, 0.50, 0.05)
        st.caption("Note: UI thresholds tweak display logic only; internal scores are shown for transparency.")

    # Button
    btn_check = st.button("Check Authorship", type="primary", use_container_width=True)

    # Run
    if btn_check:
        # overwrite thresholds used in classifier by temporarily monkey-patching decision logic
        old_classify = _classify_author_match

        def patched_classify(target_name: str, url: str, html_text: str) -> dict:
            res = old_classify(target_name, url, html_text)
            # Re-map decision using UI thresholds
            if res["token_coverage"] >= cov_full or res["score_ratio"] >= ratio_full:
                res["decision"] = "MATCH"
            elif res["token_coverage"] >= cov_possible or res["score_ratio"] >= ratio_possible:
                res["decision"] = "POSSIBLE_MATCH"
            else:
                res["decision"] = "NO_MATCH"
            return res

        globals()["_classify_author_match"] = patched_classify

        try:
            author = author_input.strip()
            raw_urls = [u.strip() for u in re.split(r"[\n,]", urls_input) if u.strip()]
            urls = [_strip_tracking(_normalize_url(u, "")) for u in raw_urls]

            if not author:
                st.warning("Please enter an Author Name.")
            elif not urls:
                st.warning("Please enter at least one URL.")
            else:
                with st.spinner("Verifying authorship…"):
                    results = verify_author_for_urls(author, urls)

                # Summary table
                rows = []
                for r in results:
                    rows.append({
                        "decision": r["decision"],
                        "url": r["url"],
                        "canonical_url": r["canonical_url"],
                        "domain": r["domain"],
                        "score_ratio": r["score_ratio"],
                        "token_coverage": r["token_coverage"],
                        "evidence_sources": "; ".join(sorted(set(r["evidence_sources"]))),
                        "top_candidate": (r["found_candidates"][0]["candidate"]
                                          if r["found_candidates"] else ""),
                    })
                df = pd.DataFrame(rows)
                st.markdown("#### Results")
                st.dataframe(df, use_container_width=True)

                # Download
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(results, indent=2, ensure_ascii=False),
                        file_name="author_verification_results.json",
                        mime="application/json",
                    )
                with c2:
                    st.download_button(
                        "Download CSV",
                        data=df.to_csv(index=False),
                        file_name="author_verification_results.csv",
                        mime="text/csv",
                    )

                # Details
                for r in results:
                    with st.expander(f"Details — {r['url']}"):
                        st.write(f"**Decision:** {r['decision']}")
                        st.write(f"**Canonical:** {r['canonical_url']}")
                        st.write(f"**Domain:** {r['domain']}")
                        st.write(f"**Fuzzy ratio:** {r['score_ratio']}  |  **Token coverage:** {r['token_coverage']}")
                        if r["evidence_sources"]:
                            st.write("**Evidence sources:** " + ", ".join(sorted(set(r["evidence_sources"]))))
                        if r["found_candidates"]:
                            st.write("**Candidates:**")
                            st.dataframe(pd.DataFrame(r["found_candidates"]))
                        if r["notes"]:
                            st.info(r["notes"])

        finally:
            # restore original classifier
            globals()["_classify_author_match"] = old_classify

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built with requests + BeautifulSoup. If a site becomes fully client-rendered or blocks bots, "
           "slug heuristics and partial evidence are used; upload HTML or use a headful browser if needed.")
