import requests
from bs4 import BeautifulSoup

def fetch_directory(url):
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    experts = {}
    for block in soup.select("div.expert-block"):  # adjust selectors as needed
        name_tag = block.select_one(".expert-name")
        link_tags = block.select("a")
        if name_tag:
            name = name_tag.get_text(strip=True)
            urls = [a["href"] for a in link_tags if a.get("href")]
            experts[name.lower()] = urls
    return experts

directory = fetch_directory("https://www.mlforseo.com/experts/")

# Later, use `directory` when processing submitted URLs.
