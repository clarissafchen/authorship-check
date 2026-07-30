# Writer Attribution Checker

Verify whether web content is genuinely authored by a specific writer using bylines, metadata, JSON-LD, and other attribution signals.

**🚀 Live App:** https://writer-attribution-check.streamlit.app/

## Features

- Verify a single author against up to 5 content URLs
- Bulk verify author directories and expert listings
- Automatically scrape author pages and submitted content
- Detect byline, JSON-LD, meta author tags, and other attribution signals
- Exclude profile/About pages from positive matches
- Explain why each URL matched or failed
- Export results as CSV, Excel, or JSON

## How to Use

### Quick Verification
1. Enter an author's name.
2. Paste up to 5 article URLs.
3. Click **Verify author URLs**.

### Bulk Verification
1. Enter one or more author directory or expert listing URLs.
2. Choose a scraping mode (Heuristic, CSS Selectors, or Marker Text).
3. Scrape the directory.
4. Click **Verify now** to analyze every discovered URL.

## Tech Stack

- Streamlit
- BeautifulSoup
- Requests
- Playwright (optional)
- pandas
- lxml
- JSON-LD parsing
