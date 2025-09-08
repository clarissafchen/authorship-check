import requests
from bs4 import BeautifulSoup

def scrape_expert_directory(url):
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')

    expert_data = {}
    for expert in soup.select(".expert-item"):  # Adjust this selector based on actual HTML
        name_tag = expert.select_one(".expert-name")
        link_tags = expert.select("a")

        if name_tag:
            name = name_tag.get_text(strip=True)
            urls = [a["href"] for a in link_tags if a.get("href") and a["href"].startswith("http")]
            expert_data[name.lower()] = urls
    return expert_data
