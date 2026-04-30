"""
scrape_gospel.py
Scrapes the complete Gospel of Sri Ramakrishna from
https://www.ramakrishnavivekananda.info/gospel/
and saves it as docs/gospel_of_sri_ramakrishna.txt
"""

import time
import requests
from bs4 import BeautifulSoup

BASE    = "https://www.ramakrishnavivekananda.info/gospel/"
OUTPUT  = "docs/gospel_of_sri_ramakrishna.txt"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ── All pages to scrape in order ─────────────────────────────────────────────
def get_all_urls():
    urls = []

    # Front matter (single pages)
    for page in ["foreword.htm", "preface.htm", "mahendranath_gupta.htm",
                 "introduction/introduction.htm"]:
        urls.append(BASE + page)

    # Mahasamadhi page in introduction (linked from volume 1 TOC)
    urls.append(BASE + "introduction/mahasamadhi.htm")

    # Volume 1 chapters
    r = requests.get(BASE + "volume_1/volume_1.htm", headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.endswith(".htm") and "/" not in h and h not in ("../gospel.htm",):
            full = BASE + "volume_1/" + h
            if full not in urls:
                urls.append(full)

    # Volume 2 chapters
    r = requests.get(BASE + "volume_2/volume_2.htm", headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.endswith(".htm") and "/" not in h and h not in ("../gospel.htm",):
            full = BASE + "volume_2/" + h
            if full not in urls:
                urls.append(full)

    # Back matter
    for page in ["appendix_a.htm", "appendix_b.htm",
                 "chronology.htm", "glossary.htm"]:
        urls.append(BASE + page)

    return urls


# ── Scrape a single page ──────────────────────────────────────────────────────
def scrape_page(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Remove nav links, scripts, styles
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    # Remove the small navigation arrows (←/→)
    for a in soup.find_all("a"):
        if a.get_text(strip=True) in ("←", "→", "Home"):
            a.decompose()

    # Get the main content — try <article>, then <body>
    main = soup.find("article") or soup.find("body")
    if main:
        text = main.get_text(separator="\n")
    else:
        text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [l.rstrip() for l in text.splitlines()]
    # Collapse 3+ blank lines to 2
    cleaned = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    urls = get_all_urls()
    # Deduplicate while preserving order
    seen, unique = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    urls = unique

    print(f"Found {len(urls)} pages to scrape\n")

    all_text = [
        "THE GOSPEL OF SRI RAMAKRISHNA",
        "By Swami Nikhilananda",
        "Source: https://www.ramakrishnavivekananda.info/gospel/",
        "=" * 70,
        "",
    ]

    for i, url in enumerate(urls, 1):
        slug = url.replace(BASE, "").replace(".htm", "")
        print(f"[{i:>3}/{len(urls)}] {slug}")
        try:
            text = scrape_page(url)
            if text:
                all_text.append(f"\n{'=' * 70}\n{slug.upper()}\n{'=' * 70}\n")
                all_text.append(text)
                print(f"        {len(text):,} chars")
            else:
                print(f"        (empty)")
        except Exception as e:
            print(f"        ERROR: {e}")
        time.sleep(0.3)   # polite delay

    full = "\n".join(all_text)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(full)

    print(f"\nSaved to {OUTPUT}")
    print(f"Total: {len(full):,} characters")


if __name__ == "__main__":
    main()
