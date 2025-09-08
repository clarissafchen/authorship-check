import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def extract_author_info(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
    except Exception as e:
        return {
            "url": url,
            "author": None,
            "match": False,
            "reason": f"Failed to fetch: {e}"
        }

    soup = BeautifulSoup(res.text, 'html.parser')

    # Try common selectors
    meta_author = soup.find('meta', attrs={'name': 'author'}) or soup.find('meta', attrs={'property': 'article:author'})
    og_author = soup.find('meta', attrs={'property': 'og:article:author'})
    byline = soup.select_one('.author, .byline, [rel="author"], .post-author')

    # Extract values
    author_name = (
        meta_author['content'] if meta_author and 'content' in meta_author.attrs else None
    ) or (
        og_author['content'] if og_author and 'content' in og_author.attrs else None
    ) or (
        byline.get_text(strip=True) if byline else None
    )

    return {
        "url": url,
        "author": author_name
    }


def check_authorship(applicant_name, urls):
    results = []
    for url in urls:
        info = extract_author_info(url)

        # Fix starts here — skip if extract_author_info returned None
        if info is None:
            results.append({
                "url": url,
                "author": None,
                "match": False,
                "reason": "Extraction failed (returned None)"
            })
            continue

        # Defensive defaults
        author_raw = info.get("author") or ""
        normalized_author = author_raw.lower().strip()
        normalized_applicant = applicant_name.lower().strip()

        info["match"] = normalized_applicant in normalized_author if normalized_author else False
        info["reason"] = (
            "Match" if info["match"] else
            "Author not attributed" if not author_raw else
            f"Attributed to different author: {author_raw}"
        )
        results.append(info)

    return results


# Example use:
applicant = "Chai Fisher"
submitted_urls = [
    "https://theseocommunity.com/about-us/chai-fisher",
    "https://seoarmy.co/seo-vs-geo-why-you-dont-have-to-choose-sides/",
    "https://theseocommunity.com/resources/blog/campfire-newsletter-9%3A-on-entropy",
    "https://theseocommunity.com/resources/blog/ai-prompts-for-job-applications"
]

results = check_authorship(applicant, submitted_urls)
for r in results:
    print(f"\nURL: {r['url']}\n→ Author: {r.get('author')}\n→ Match: {r['match']}\n→ Reason: {r['reason']}")
