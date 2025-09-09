# Streamlit app:
#   - Tab 1: Scrape https://www.mlforseo.com/experts/ grid (requests + BeautifulSoup)
#   - Tab 2: Verify authorship of submitted content URLs against a provided author name
#
# Notes:
# - Author check gathers evidence from JSON-LD, meta tags, visible bylines, domain adapters
# - LinkedIn Pulse often blocks bots; we add slug heuristics + JSON-LD regex fallback
# - Matching uses normalized tokens + fuzzy ratio (difflib) to classify MATCH/POSSIBLE/NO_MATCH

import json
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

# ---------- Constants
DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
REQ_TIMEOUT = 15

# Try to import Playwright lazily (optional).
PLAYWRIGHT_IMPORTED = False
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError  # type: ignore
    PLAYWRIGHT_IMPORTED = True
except Exception:
    PLAYWRIGHT_IMPORTED = False

# ---------- Small helpers
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def safe_text(el) -> str:
    return norm_space(el.get_text(" ", strip=True)) if el else ""

def to_ascii_lower(s: str) -> str:
    try:
        import unicodedata
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    except Exception:
        pass
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_set(name: str) -> List[str]:
    return [t for t in re.split(r"[\s\-_/]+", to_ascii_lower(name)) if t]

def dl_button_bytes(label: str, data: bytes, file_name: str, mime: str):
    st.download_button(label, data=data, file_name=file_name, mime=mime)

@dataclass
class HTTPDiag:
    requested_url: str
    final_url: str
    status_code: int
    elapsed_ms: int
    length: int
    engine: str

@dataclass
class ParseDiag:
    parser: str
    details: Dict[str, Any]

# ---------- Networking
def fetch_html_requests(url: str, headers: Optional[dict] = None) -> Tuple[str, HTTPDiag]:
    t0 = time.time()
    r = requests.get(
        url,
        headers=headers or {"User-Agent": DEFAULT_UA, "Accept-Language": "en-US,en;q=0.9"},
        timeout=REQ_TIMEOUT,
    )
    html = r.text
    diag = HTTPDiag(
        requested_url=url,
        final_url=str(r.url),
        status_code=r.status_code,
        elapsed_ms=int((time.time() - t0) * 1000),
        length=len(html or ""),
        engine="requests",
    )
    return html, diag

# --- one-time install helper
def install_playwright_browsers() -> Tuple[bool, str]:
    if not PLAYWRIGHT_IMPORTED:
        return False, "Playwright not imported"
    import subprocess, sys
    try:
        cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (proc.returncode == 0)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return ok, out
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def fetch_html_playwright(url: str, wait_selector: Optional[str] = None, timeout_ms: int = 10000) -> Tuple[Optional[str], Optional[HTTPDiag], Optional[str]]:
    if not PLAYWRIGHT_IMPORTED:
        return None, None, "playwright_not_available"

    def _launch_and_get() -> Tuple[str, HTTPDiag]:
        t0 = time.time()
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=DEFAULT_UA, locale="en-US")
            page = ctx.new_page()
            resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            selector = wait_selector or "article.expert-card, .experts-grid article, .experts-grid [data-cats]"
            try:
                page.wait_for_selector(selector, timeout=timeout_ms)
            except PWTimeoutError:
                pass
            html = page.content()
            final_url = page.url
            status = resp.status if resp else 200
            browser.close()
        diag = HTTPDiag(
            requested_url=url,
            final_url=final_url,
            status_code=int(status or 0),
            elapsed_ms=int((time.time() - t0) * 1000),
            length=len(html or ""),
            engine="playwright",
        )
        return html, diag

    try:
        html, diag = _launch_and_get()
        return html, diag, None
    except Exception as e:
        msg = str(e)
        needs_install = ("Executable doesn't exist" in msg) or ("playwright was just installed" in msg.lower())
        if needs_install:
            ok, out = install_playwright_browsers()
            if not ok:
                return None, None, f"playwright_install_failed: {out.strip()}"
            try:
                html, diag = _launch_and_get()
                return html, diag, None
            except Exception as e2:
                return None, None, f"{type(e2).__name__}: {e2}"
        return None, None, f"{type(e).__name__}: {e}"

# ---------- Parsers (Grid + JSON-LD Person)
def parse_grid_cards(html: str) -> Tuple[List[Dict[str, Any]], ParseDiag]:
    soup = BeautifulSoup(html, "html.parser")

    cards = soup.select("article.expert-card")
    if not cards:
        cards = soup.select(".experts-grid article, .experts-grid [data-cats]")

    results: List[Dict[str, Any]] = []

    for c in cards:
        img = c.select_one(".photo-wrap img") or c.select_one("img")
        photo = img.get("src") if img else None

        name_a = c.select_one("h3.name a")
        author_text = safe_text(name_a) if name_a else safe_text(c.select_one("h3.name"))
        profile_url = name_a.get("href") if name_a else None

        site_a = c.select_one(".icons a.site, .icons .icon-link.site, a.icon-link.site")
        website = site_a.get("href") if site_a else None

        chips = [safe_text(x) for x in c.select(".chips-row .chip, .chip[data-chip]")]
        chips = [x for x in chips if x]

        reason = safe_text(c.select_one(".reason"))

        resources = []
        for a in c.select(".resources-list a.res-pill, .resources a"):
            href = a.get("href")
            label = safe_text(a.select_one(".res-text")) or safe_text(a)
            fav = None
            fav_img = a.select_one("img.res-favicon")
            if fav_img:
                fav = fav_img.get("src")
            if href:
                resources.append({"href": href, "label": label, "favicon": fav})

        if author_text:
            results.append(
                {
                    "author": author_text,           # <-- renamed from "name"
                    "profile_url": profile_url,
                    "website": website,
                    "photo": photo,
                    "categories": chips,
                    "reason": reason,
                    "resources": resources,
                }
            )

    diag = ParseDiag(
        parser="mlforseo_grid_css",
        details={"cards_found": len(cards), "experts_parsed": len(results)},
    )
    return results, diag

def parse_schema_person(html: str) -> Tuple[List[Dict[str, Any]], ParseDiag]:
    soup = BeautifulSoup(html, "html.parser")
    people: List[Dict[str, Any]] = []

    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    hits = 0
    for s in scripts:
        try:
            data = json.loads(s.string or "{}")
        except Exception:
            continue

        def _collect(obj):
            nonlocal people, hits
            if isinstance(obj, list):
                for it in obj:
                    _collect(it)
            elif isinstance(obj, dict):
                t = obj.get("@type")
                if t == "Person" or (isinstance(t, list) and "Person" in t):
                    hits += 1
                    person = {
                        "author": obj.get("name"),
                        "jobTitle": obj.get("jobTitle"),
                        "url": obj.get("url"),
                        "sameAs": obj.get("sameAs"),
                        "image": obj.get("image"),
                    }
                    people.append(person)
                for v in obj.values():
                    _collect(v)

        _collect(data)

    diag = ParseDiag("schema_person", {"persons_found": len(people), "jsonld_hits": hits})
    return people, diag

def scrape_experts(url: str, prefer_js: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    html, http_req = fetch_html_requests(url)
    grid, d_grid = parse_grid_cards(html)
    if grid:
        return grid, {"http": asdict(http_req), "parse": asdict(d_grid)}, None

    persons, d_schema = parse_schema_person(html)
    if persons:
        return persons, {"http": asdict(http_req), "parse": {"parser": "schema_person", **d_schema.details}}, None

    if prefer_js:
        js_html, http_js, js_err = fetch_html_playwright(url, wait_selector="article.expert-card, .experts-grid [data-cats]")
        if js_html and http_js:
            grid2, d_grid2 = parse_grid_cards(js_html)
            if grid2:
                return grid2, {"http": asdict(http_js), "parse": asdict(d_grid2)}, None
            persons2, d_schema2 = parse_schema_person(js_html)
            if persons2:
                return persons2, {"http": asdict(http_js), "parse": {"parser": "schema_person", **d_schema2.details}}, None
            return [], {"http": asdict(http_js), "parse": {"parser": "mlforseo_grid_css", "cards_found": 0, "experts_parsed": 0}}, None
        else:
            diag = {"http": asdict(http_req), "parse": {"auto": {
                "mlforseo_grid": {"parser": "mlforseo_grid_css", "cards_found": 0, "experts_parsed": 0},
                "schema_person": {"parser": "schema_person", "persons_found": 0, "jsonld_hits": 0}
            }}}
            if js_err:
                diag["js_error"] = js_err
            return [], diag, js_err

    return [], {
        "http": asdict(http_req),
        "parse": {"mlforseo_grid_css": {"cards_found": 0, "experts_parsed": 0}}
    }, None

# ---------- Author verification (shared with Tab 2)
def guess_author_from_meta_and_jsonld(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []

    for sel in [
        'meta[name="author"]',
        'meta[property="article:author"]',
        'meta[name="byl"]',
        'meta[property="og:author"]',
        'meta[name="parsely-author"]',
    ]:
        for m in soup.select(sel):
            v = (m.get("content") or "").strip()
            if v:
                candidates.append(v)

    for sel in ["[rel='author']", ".byline a", ".author a", ".author-name", '[itemprop="author"] [itemprop="name"]']:
        for el in soup.select(sel):
            t = safe_text(el)
            if t:
                candidates.append(t)

    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for s in scripts:
        try:
            data = json.loads(s.string or "{}")
        except Exception:
            continue

        def _collect(obj):
            if isinstance(obj, list):
                for it in obj:
                    _collect(it)
            elif isinstance(obj, dict):
                t = obj.get("@type")
                if t == "Person" or (isinstance(t, list) and "Person" in t):
                    nm = obj.get("name")
                    if nm:
                        candidates.append(str(nm))
                for v in obj.values():
                    _collect(v)

        _collect(data)

    seen = set()
    uniq = []
    for c in candidates:
        key = to_ascii_lower(c)
        if key and key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def slug_tokens_from_url(url: str) -> List[str]:
    try:
        p = urlparse(url)
        path = unquote(p.path or "")
        path = re.sub(r"[^A-Za-z0-9/_\-]", " ", path)
        toks = token_set(path)
        return toks
    except Exception:
        return []

def linkedin_pulse_match(author: str, url: str) -> Tuple[bool, float, str, Optional[str]]:
    a_toks = [t for t in token_set(author) if t]
    if len(a_toks) < 1:
        return False, 0.0, "no_author_tokens", None

    path_tokens = slug_tokens_from_url(url)
    if not path_tokens:
        return False, 0.0, "no_path_tokens", None

    need = {a_toks[0]}
    if len(a_toks) >= 2:
        need.add(a_toks[-1])

    present = need.issubset(set(path_tokens))
    if present:
        return True, 0.7, "linkedin_slug_match", None
    return False, 0.0, "linkedin_slug_no_match", None

def verify_authorship(author: str, url: str) -> Dict[str, Any]:
    domain = (urlparse(url).netloc or "").lower()
    norm_author = to_ascii_lower(author)
    author_tokens = token_set(author)

    if "linkedin.com" in domain and "/pulse/" in url:
        ok, conf, method, _ = linkedin_pulse_match(author, url)
        result = {
            "url": url,
            "declared_author": author,
            "detected_author": author if ok else None,
            "method": method,
            "confidence": conf,
            "match": bool(ok),
            "note": "Slug-based verification for LinkedIn Pulse",
            "domain": domain
        }
        try:
            html, _ = fetch_html_requests(url)
            cands = guess_author_from_meta_and_jsonld(html)
            for cand in cands:
                if to_ascii_lower(cand) == norm_author:
                    result["detected_author"] = cand
                    result["method"] = "meta/jsonld"
                    result["confidence"] = 0.95
                    result["match"] = True
                    result["note"] = "Confirmed via meta/JSON-LD"
                    break
        except Exception:
            pass
        return result

    try:
        html, _ = fetch_html_requests(url)
        cands = guess_author_from_meta_and_jsonld(html)
    except Exception as e:
        return {
            "url": url,
            "declared_author": author,
            "detected_author": None,
            "method": "fetch_error",
            "confidence": 0.0,
            "match": False,
            "note": f"{type(e).__name__}: {e}",
            "domain": domain
        }

    detected = None
    method = None
    conf = 0.0

    for cand in cands:
        if to_ascii_lower(cand) == norm_author:
            detected = cand
            method = "meta/jsonld"
            conf = 0.95
            break

    if not detected:
        path_toks = slug_tokens_from_url(url)
        need = set()
        if author_tokens:
            need.add(author_tokens[0])
            if len(author_tokens) > 1:
                need.add(author_tokens[-1])

        if need and need.issubset(set(path_toks)):
            detected = author
            method = "url_slug"
            conf = 0.7

    return {
        "url": url,
        "declared_author": author,
        "detected_author": detected,
        "method": method or "not_found",
        "confidence": conf,
        "match": bool(detected),
        "note": "" if detected else "No author detected from meta/JSON-LD or slug",
        "domain": domain
    }

def evaluate_authorship_for_resources(author: str, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate each resource link for authorship; return summary + per-link details."""
    evaluations = []
    authored = 0
    total = 0

    for r in resources or []:
        url = r.get("href")
        label = r.get("label")
        if not url:
            continue
        total += 1
        res = verify_authorship(author, url)
        res["label"] = label
        res["favicon"] = r.get("favicon")
        if res.get("match"):
            authored += 1
        evaluations.append(res)

    rate = (authored / total) if total else 0.0
    return {
        "resources_authored": authored,
        "resources_total": total,
        "authorship_rate": rate,
        "evaluations": evaluations
    }

# ---------- UI
st.set_page_config(page_title="MLforSEO Tools", page_icon="🧰", layout="wide")
st.title("MLforSEO Tools")

tab1, tab2 = st.tabs(["🔎 Scrape Experts Grid", "📝 Check Author-Submitted Content"])

with tab1:
    st.subheader("Scrape the experts grid and verify resource authorship.")

    col_l, col_r = st.columns([2, 1])
    with col_l:
        url = st.text_input("Experts page URL", value="https://www.mlforseo.com/experts/")
    with col_r:
        show_photos = st.checkbox("Show headshots", value=False)

    colx, coly, colz = st.columns([1,1,1])
    with colx:
        use_js = st.checkbox("Render JavaScript (headless browser)", value=True)
    with coly:
        st.caption("Playwright imported: **{}**".format("✅" if PLAYWRIGHT_IMPORTED else "❌"))
    with colz:
        do_eval = st.checkbox("Evaluate authorship of resource links", value=True)

    if use_js and PLAYWRIGHT_IMPORTED:
        if st.button("🔧 One-time setup: install headless Chromium"):
            with st.spinner("Installing Playwright browser (chromium)…"):
                ok, out = install_playwright_browsers()
            st.toast("Playwright browser installed." if ok else "Playwright install failed.", icon="✅" if ok else "❌")
            st.expander("Install output").code(out)

    if st.button("Scrape Experts", type="primary"):
        with st.spinner("Fetching and parsing…"):
            data, diag, js_err = scrape_experts(url, prefer_js=use_js)

        total_experts = len(data)
        if total_experts:
            st.success(f"Found {total_experts} expert(s).")
        else:
            st.info("Found 0 experts.")

        # Optionally evaluate authorship for each expert's resources
        per_resource_rows = []
        if data and do_eval:
            prog = st.progress(0.0, text="Evaluating authorship for resource links…")
            for i, item in enumerate(data):
                author = item.get("author")
                resources = item.get("resources") or []
                eval_pack = evaluate_authorship_for_resources(author, resources)
                # attach summary back on the expert item
                item["resources_authored"] = eval_pack["resources_authored"]
                item["resources_total"] = eval_pack["resources_total"]
                item["authorship_rate"] = eval_pack["authorship_rate"]
                item["resource_evaluations"] = eval_pack["evaluations"]

                # flatten per-resource evaluation rows for separate table/export
                for ev in eval_pack["evaluations"]:
                    per_resource_rows.append({
                        "author": author,
                        "url": ev.get("url"),
                        "domain": ev.get("domain"),
                        "label": ev.get("label"),
                        "method": ev.get("method"),
                        "confidence": ev.get("confidence"),
                        "match": ev.get("match"),
                        "detected_author": ev.get("detected_author"),
                        "note": ev.get("note"),
                    })
                if total_experts:
                    prog.progress((i+1)/total_experts)
            prog.empty()

        if data:
            # Build the main experts table
            flat_rows = []
            for item in data:
                flat_rows.append({
                    "author": item.get("author"),
                    "profile_url": item.get("profile_url"),
                    "website": item.get("website"),
                    "photo": item.get("photo"),
                    "categories": ", ".join(item.get("categories") or []),
                    "reason": item.get("reason"),
                    "resources": ", ".join([r.get("href") for r in item.get("resources", [])]),
                    "resources_authored": item.get("resources_authored", None),
                    "resources_total": item.get("resources_total", None),
                    "authorship_rate": item.get("authorship_rate", None),
                })
            df = pd.DataFrame(flat_rows)

            if show_photos and "photo" in df.columns:
                def _img_md(u): return f"![]({u})" if u else ""
                df_disp = df.copy()
                df_disp["photo"] = df_disp["photo"].map(_img_md)
                st.dataframe(df_disp, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)

            # Downloads for experts
            jbytes = json.dumps(data, indent=2).encode("utf-8")
            dl_button_bytes("Download Experts JSON", jbytes, "experts.json", "application/json")
            dl_button_bytes("Download Experts CSV", df.to_csv(index=False).encode("utf-8"), "experts.csv", "text/csv")

            # Per-resource evaluations table/export (if any)
            if per_resource_rows:
                st.markdown("**Per-resource authorship checks**")
                df_ev = pd.DataFrame(per_resource_rows)
                st.dataframe(df_ev, use_container_width=True)
                dl_button_bytes("Download Resource Checks JSON", json.dumps(per_resource_rows, indent=2).encode("utf-8"),
                                "resource_checks.json", "application/json")
                dl_button_bytes("Download Resource Checks CSV", df_ev.to_csv(index=False).encode("utf-8"),
                                "resource_checks.csv", "text/csv")

        with st.expander("Diagnostics"):
            st.code(json.dumps(diag, indent=2))
            if js_err and use_js:
                st.warning(f"JS rendering error: {js_err}")
                if "playwright_install_failed" in js_err or "not_available" in js_err:
                    st.info("Click the **🔧 One-time setup** button above, then try again.")

with tab2:
    st.subheader("Verify an author name against one or more URLs.")

    a_col, u_col = st.columns([1, 2])
    with a_col:
        author_name = st.text_input("Author Name", value="Aimee Jurenka")
    with u_col:
        urls_raw = st.text_area(
            "Content URLs (comma-separated)",
            value="https://www.linkedin.com/pulse/buzzword-betty-vol-1-vector-embeddings-cosine-semantic-jurenka-8iqwc/?trackingId=7pahQErMQzmJThdkY8KSsw%3D%3D, https://www.linkedin.com/pulse/buzzword-betty-vol-6-agentic-agents-mcps-aimee-jurenka-oekoc/?trackingId=rDFpw%2F2tQCa%2BYxTdLJxo3Q%3D%3D"
        )

    if st.button("Run Checks", type="primary"):
        urls = [u.strip() for u in urls_raw.split(",") if u.strip()]
        rows = [verify_authorship(author_name, u) for u in urls]

        df = pd.DataFrame(rows, columns=["url","declared_author","detected_author","method","confidence","match","note","domain"])
        st.dataframe(df, use_container_width=True)

        ok = sum(1 for r in rows if r.get("match"))
        if ok:
            st.success(f"{ok}/{len(rows)} URLs matched the declared author.")
        else:
            st.error("No matches found.")

        jbytes = json.dumps(rows, indent=2).encode("utf-8")
        dl_button_bytes("Download JSON", jbytes, "author_checks.json", "application/json")
        dl_button_bytes("Download CSV", df.to_csv(index=False).encode("utf-8"), "author_checks.csv", "text/csv")
