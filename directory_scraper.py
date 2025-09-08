from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def scrape_expert_directory(url="https://www.mlforseo.com/experts/"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_selector("article.expert-card", timeout=1000000)  # Wait for JS to render
        content = page.content()
        browser.close()

    soup = BeautifulSoup(content, "html.parser")
    expert_data = {}

    cards = soup.select("article.expert-card")
    for card in cards:
        name_tag = card.select_one("h3.name a")
        name = name_tag.get_text(strip=True) if name_tag else None

        urls = []
        for link in card.select("ul.resources-list a.res-pill"):
            href = link.get("href")
            if href and href.startswith("http"):
                urls.append(href)

        if name and urls:
            expert_data[name.lower()] = urls

    return expert_data
