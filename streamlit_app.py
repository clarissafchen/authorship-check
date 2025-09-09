# Streamlit app:
#   - Tab 1: Scrape https://www.mlforseo.com/experts/ grid (requests + BeautifulSoup)
#   - Tab 2: Verify authorship of submitted content URLs against a provided author name
#
# Notes:
# - Author check gathers evidence from JSON-LD, meta tags, visible bylines, domain adapters
# - LinkedIn Pulse often blocks bots; we add slug heuristics + JSON-LD regex fallback
# - Matching uses normalized tokens + fuzzy ratio (difflib) to classify MATCH/POSSIBLE/NO_MATCH

# streamlit_app.py
import re
import csv
import io
import json
import time
import difflib
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
import streamlit as st

APP_TITLE = "MLforSEO Tools"
DEFAULT_EXPERTS_URL = "https://www.mlforseo.com/experts/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# -------------------- Utilities

def normalize_name(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
    s = re.sub(r"[\u2019'’`]", "", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def fuzzy_same_author(a: str, b: str, threshold=0.84) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    if difflib.SequenceMatcher(None, na, nb).ratio() >= threshold:
        return True
    sa = " ".join(sorted(na.split()))
    sb = " ".join(sorted(nb.split()))
    return difflib.SequenceMatcher(None, sa, sb).ratio() >= threshold

def to_csv_bytes(rows, fieldnames):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})
    return buf.getvalue().encode("utf-8")

def canonicalize(u: str) -> str:
    try:
        p = urlparse(u)
        qs = re.sub(r"(^|&)(utm_[^=]+|fbclid|gclid|trackingId)=[^&]*", "", p.query)
        qs = re.sub(r"&&+", "&", qs).strip("&")
        return urlunparse((p.scheme, p.netloc, p.path, "", qs, ""))
    except Exception:
        return u

@st.cache_data(ttl=600, show_spinner=False)
def fetch_html(url: str):
    t0 = time.time()
    resp = requests.get(url, headers=HEADERS, timeout=25, allow_redirects=True)
    elapsed = round((time.time() - t0) * 1000)
    meta = {
        "requested_url": url,
        "final_url": resp.url,
        "status_code": resp.status_code,
        "elapsed_ms": elapsed,
        "length": len(resp.text or ""),
    }
    resp.raise_for_status()
    return resp.text, meta

# -------------------- Parsers

def parse_mlforseo_css(html: str):
    soup = BeautifulSoup(html, "html.parser")
    grid = soup.select_one("#experts-grid.experts-grid")
    cards = grid.select("article.expert-card") if grid else []

    results = []
    for art in cards:
        name_el = art.select_one(".title-row .name a") or art.select_one(".title-row .name")
        name = name_el.get_text(strip=True) if name_el else ""

        profile_url = ""
        if name_el and name_el.name == "a" and name_el.has_attr("href"):
            profile_url = name_el["href"].strip()

        site_el = art.select_one(".icons a.site[href]")
        website = site_el["href"].strip() if site_el else ""

        img_el = art.select_one(".photo-wrap img[src]")
        photo = img_el["src"].strip() if img_el else ""

        chips = [c.get("data-chip", "").strip() for c in art.select(".chips-row .chip") if c.get("data-chip")]
        categories = [c for c in chips if c]

        reason_el = art.select_one("p.reason")
        reason = reason_el.get_text(strip=True) if reason_el else ""

        resources = []
        for a in art.select(".resources .resources-list a.res-pill[href]"):
            href = a["href"].strip()
            label_el = a.select_one(".res-text")
            label = label_el.get_text(strip=True) if label_el else urlparse(href).netloc
            resources.append({"url": href, "label": label})

        results.append(
            {
                "name": name,
                "profile_url": profile_url,
                "website": website,
                "photo": photo,
                "categories": categories,
                "reason": reason,
                "resources": resources,
            }
        )
    diag = {"parser": "mlforseo_css", "grid_found": grid is not None, "cards_found": len(cards)}
    return results, diag

def _flatten_json(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from _flatten_json(x)
    elif isinstance(obj, dict) and "@graph" in obj:
        yield from _flatten_json(obj["@graph"])
    else:
        yield obj

def parse_schema_persons(html: str):
    """Domain-agnostic: read JSON-LD Person items."""
    soup = BeautifulSoup(html, "html.parser")
    persons = []
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        raw = tag.string or ""
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for node in _flatten_json(data):
            if not isinstance(node, dict):
                continue
            t = node.get("@type")
            if isinstance(t, list):
                is_person = any(tt.lower() == "person" for tt in t if isinstance(tt, str))
            else:
                is_person = isinstance(t, str) and t.lower() == "person"
            if not is_person:
                continue

            name = node.get("name", "")
            url = node.get("url", "")
            image = node.get("image", "") or (node.get("photo") if isinstance(node.get("photo"), str) else "")
            sameas = node.get("sameAs", [])
            if isinstance(sameas, str):
                sameas = [sameas]
            # pick a good profile if available
            profile_url = ""
            for s in sameas or []:
                if "linkedin.com" in s or "twitter.com" in s or "x.com" in s:
                    profile_url = s
                    break
            if not profile_url:
                profile_url = url

            # categories / topics
            cats = []
            for key in ("knowsAbout", "keywords", "areasOfExpertise", "areasServed"):
                v = node.get(key)
                if isinstance(v, list):
                    cats.extend([str(x) for x in v])
                elif isinstance(v, str):
                    cats.extend([x.strip() for x in re.split(r"[;,]", v) if x.strip()])
            cats = list(dict.fromkeys([c for c in cats if c]))

            # resources: use sameAs as a first-class list
            resources = [{"url": s, "label": urlparse(s).netloc or s} for s in (sameas or [])]

            reason = node.get("description", "")
            persons.append(
                {
                    "name": name,
                    "profile_url": profile_url,
                    "website": url,
                    "photo": image,
                    "categories": cats,
                    "reason": reason,
                    "resources": resources,
                }
            )
    diag = {"parser": "schema_person", "persons_found": len(persons)}
    return persons, diag

def parse_custom_css(html: str, sel: dict):
    """
    Flexible CSS parser. sel keys:
      card, name, name_link (optional), profile_link, website_link, photo_img,
      category_chip, reason, resource_link, resource_label
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(sel.get("card") or "")
    out = []
    for art in cards:
        def pick_text(selector):
            el = art.select_one(selector) if selector else None
            return el.get_text(strip=True) if el else ""

        def pick_href(selector):
            el = art.select_one(selector) if selector else None
            return el["href"].strip() if el and el.has_attr("href") else ""

        def pick_src(selector):
            el = art.select_one(selector) if selector else None
            return el["src"].strip() if el and el.has_attr("src") else ""

        name = pick_text(sel.get("name") or "")
        if not name and sel.get("name_link"):
            el = art.select_one(sel["name_link"])
            name = el.get_text(strip=True) if el else ""

        profile_url = pick_href(sel.get("profile_link") or sel.get("name_link") or "")
        website = pick_href(sel.get("website_link") or "")
        photo = pick_src(sel.get("photo_img") or "")

        cats = []
        chip_sel = sel.get("category_chip")
        if chip_sel:
            for chip in art.select(chip_sel):
                # Try data-chip then text
                cats.append(chip.get("data-chip", "").strip() or chip.get_text(strip=True))
        cats = [c for c in cats if c]

        reason = pick_text(sel.get("reason") or "")

        resources = []
        link_sel = sel.get("resource_link")
        if link_sel:
            for a in art.select(link_sel):
                if not a.has_attr("href"):
                    continue
                href = a["href"].strip()
                label = ""
                lab_sel = sel.get("resource_label")
                if lab_sel:
                    lab_el = a.select_one(lab_sel)
                    label = lab_el.get_text(strip=True) if lab_el else ""
                if not label:
                    label = urlparse(href).netloc or href
                resources.append({"url": href, "label": label})

        out.append(
            {
                "name": name,
                "profile_url": profile_url,
                "website": website,
                "photo": photo,
                "categories": cats,
                "reason": reason,
                "resources": resources,
            }
        )
    diag = {"parser": "custom_css", "cards_found": len(cards)}
    return out, diag

# -------------------- Author extraction (Tab 2)

@st.cache_data(ttl=600, show_spinner=False)
def fetch_article(url: str):
    u = canonicalize(url)
    r = requests.get(u, headers=HEADERS, timeout=25)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    authors = []

    # JSON-LD
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        for node in _flatten_json(data):
            if not isinstance(node, dict):
                continue
            a = node.get("author")
            if isinstance(a, dict) and a.get("name"):
                authors.append(a["name"])
            elif isinstance(a, list):
                for ai in a:
                    if isinstance(ai, dict) and ai.get("name"):
                        authors.append(ai["name"])

    # Meta fallbacks
    for attrs in [
        {"name": "author"},
        {"property": "author"},
        {"property": "article:author"},
        {"property": "og:author"},
    ]:
        m = soup.find("meta", attrs=attrs)
        if m and m.get("content"):
            authors.append(m["content"])

    # LinkedIn byline heuristics
    host = urlparse(u).netloc.lower()
    if "linkedin.com" in host:
        byline = soup.select_one("a[href*='/in/'], a[href*='/author/']")
        if byline:
            authors.append(byline.get_text(strip=True))
        txt = soup.get_text(" ", strip=True)
        m = re.search(r"\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", txt, re.I)
        if m:
            authors.append(m.group(1))
    else:
        txt = soup.get_text(" ", strip=True)
        m = re.search(r"\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", txt, re.I)
        if m:
            authors.append(m.group(1))

    # clean & dedupe
    seen = set()
    cleaned = []
    for a in authors:
        a = a.strip()
        if not a:
            continue
        k = normalize_name(a)
        if k not in seen:
            seen.add(k)
            cleaned.append(a)

    title = (soup.title.get_text(strip=True) if soup.title else "").strip()
    return {"url": u, "title": title, "authors_found": cleaned}

# -------------------- UI

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
st.title(APP_TITLE)

tab1, tab2 = st.tabs(["🔎 Scrape Directory", "📝 Check Author-Submitted Content"])

# ---------- TAB 1
with tab1:
    st.caption("Scrape any directory page. Use Auto (schema.org Persons), the MLforSEO preset, or supply your own CSS selectors.")
    url = st.text_input("Directory URL", value=DEFAULT_EXPERTS_URL)

    mode = st.radio(
        "Parsing mode",
        ["Auto (schema.org Persons)", "Preset: MLforSEO CSS", "Custom CSS selectors"],
        index=0,
        horizontal=True,
    )

    with st.expander("Custom selector profile (only used if mode = Custom CSS selectors)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            card = st.text_input("Card selector", value="article.expert-card")
            name = st.text_input("Name selector (text)", value=".title-row .name")
            name_link = st.text_input("Name link selector (href)", value=".title-row .name a")
            profile_link = st.text_input("Profile link selector (href)", value="")
        with c2:
            website_link = st.text_input("Website link selector (href)", value=".icons a.site")
            photo_img = st.text_input("Photo <img> selector (src)", value=".photo-wrap img")
            category_chip = st.text_input("Category chip selector", value=".chips-row .chip")
            reason = st.text_input("Reason selector (text)", value="p.reason")
        with c3:
            resource_link = st.text_input("Resource link selector (href)", value=".resources .resources-list a")
            resource_label = st.text_input("Resource label selector (inside link)", value=".res-text")

        custom_sel = {
            "card": card,
            "name": name,
            "name_link": name_link,
            "profile_link": profile_link,
            "website_link": website_link,
            "photo_img": photo_img,
            "category_chip": category_chip,
            "reason": reason,
            "resource_link": resource_link,
            "resource_label": resource_label,
        }

    show_heads = st.checkbox("Show headshots")

    if st.button("Scrape", type="primary"):
        try:
            html, http_info = fetch_html(url)

            results, parse_diag = [], {}
            if mode.startswith("Auto"):
                results, parse_diag = parse_schema_persons(html)
                # helpful fallback: if a page doesn't expose schema.org, try the MLforSEO CSS pattern
                if not results:
                    fallback, d2 = parse_mlforseo_css(html)
                    if fallback:
                        parse_diag = {"auto_persons": 0, "fallback": d2}
                    results = fallback
            elif mode.startswith("Preset"):
                results, parse_diag = parse_mlforseo_css(html)
            else:
                results, parse_diag = parse_custom_css(html, custom_sel)

            st.success(f"Found {len(results)} profile(s)")
            if results:
                table_rows = []
                for x in results:
                    res_preview = ", ".join(r["label"] for r in x.get("resources", [])[:3])
                    cats = ", ".join(x.get("categories", []))
                    table_rows.append(
                        {
                            "name": x.get("name",""),
                            "profile_url": x.get("profile_url",""),
                            "website": x.get("website",""),
                            "categories": cats,
                            "reason": x.get("reason",""),
                            "resources_preview": res_preview,
                            "photo": x.get("photo",""),
                        }
                    )

                if show_heads:
                    cols = st.columns(3)
                    for i, x in enumerate(results):
                        with cols[i % 3]:
                            if x.get("photo"):
                                st.image(x["photo"], use_column_width=True)
                            st.markdown(f"**{x.get('name','')}**")
                            if x.get("categories"):
                                st.caption(", ".join(x["categories"]))
                            if x.get("website"):
                                st.write(f"[Website]({x['website']})")
                            if x.get("profile_url"):
                                st.write(f"[Profile]({x['profile_url']})")
                            if x.get("reason"):
                                st.write(x["reason"])
                else:
                    st.dataframe(table_rows, use_container_width=True, hide_index=True)

                json_bytes = json.dumps(results, indent=2).encode("utf-8")
                csv_bytes = to_csv_bytes(
                    table_rows,
                    fieldnames=["name","profile_url","website","categories","reason","resources_preview","photo"]
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("Download JSON", data=json_bytes, file_name="directory.json", mime="application/json")
                with c2:
                    st.download_button("Download CSV", data=csv_bytes, file_name="directory.csv", mime="text/csv")
            else:
                st.info("No profiles parsed. Try another mode or provide custom selectors (see expander).")

            with st.expander("Diagnostics"):
                st.code(json.dumps({"http": http_info, "parse": parse_diag}, indent=2))
                st.text_area("First 1200 chars of HTML", (html[:1200] or ""), height=200)
        except Exception as e:
            st.error(f"Scrape failed: {e}")

# ---------- TAB 2: Author checker
with tab2:
    st.caption("LinkedIn Pulse-aware author verification across multiple URLs.")
    author_name = st.text_input("Author Name")
    urls_raw = st.text_area("Content URLs (comma or newline separated)", height=120)

    if st.button("Check Sources", type="secondary"):
        urls = [u.strip() for u in re.split(r"[\n,]+", urls_raw) if u.strip()]
        if not urls:
            st.warning("Please add at least one URL.")
        else:
            rows = []
            for u in urls:
                try:
                    art = fetch_article(u)
                    found = art["authors_found"]
                    verdict = any(fuzzy_same_author(author_name, a) for a in found)
                    rows.append(
                        {
                            "url": art["url"],
                            "title": art["title"],
                            "authors_detected": ", ".join(found) if found else "(none)",
                            "matches_input_author": "✅ Yes" if verdict else "❌ No",
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "url": u,
                            "title": "(error)",
                            "authors_detected": f"(error: {e})",
                            "matches_input_author": "—",
                        }
                    )
            st.dataframe(rows, use_container_width=True, hide_index=True)

            json_bytes = json.dumps(rows, indent=2).encode("utf-8")
            csv_bytes = to_csv_bytes(rows, fieldnames=["url","title","authors_detected","matches_input_author"])
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download JSON", data=json_bytes, file_name="author_checks.json", mime="application/json")
            with c2:
                st.download_button("Download CSV", data=csv_bytes, file_name="author_checks.csv", mime="text/csv")
