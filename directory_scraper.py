import requests
from bs4 import BeautifulSoup

def scrape_expert_directory(base_url="https://www.mlforseo.com/experts/"):
    res = requests.get(base_url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    expert_data = {}

    # Find all expert cards
    cards = soup.select("article.expert-card")

    for card in cards:
        # Get name
        name_tag = card.select_one("h3.name a")
        name = name_tag.get_text(strip=True) if name_tag else None

        # Get all resource URLs (ignore social/media links if needed)
        urls = []
        resource_links = card.select("ul.resources-list a.res-pill")

        for link in resource_links:
            href = link.get("href")
            if href and href.startswith("http"):
                urls.append(href)

        if name and urls:
            expert_data[name.lower()] = urls

    return expert_data
