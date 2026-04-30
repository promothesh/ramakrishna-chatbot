"""
Scrapes 'Sri Ramakrishna the Great Master' from the Zoho Learn site.
Strategy: start at the first article and click 'Next' sequentially
through all chapters — no need to enumerate TOC links up front.
"""

import time
import re
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

CHROMEDRIVER = r"C:\Users\promo\.wdm\drivers\chromedriver\win64\147.0.7727.117\chromedriver-win32\chromedriver.exe"
BASE_URL     = "https://englishbooks.rkmm.org"
START_URL    = f"{BASE_URL}/s/lsr/m/sri-ramakrishna-the-great-master/a/cover"
OUTPUT_FILE  = "docs/sri_ramakrishna_great_master.txt"

os.makedirs("docs", exist_ok=True)

# ── Chrome setup ─────────────────────────────────────────────────────────────
options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)

driver = webdriver.Chrome(service=Service(CHROMEDRIVER), options=options)
wait   = WebDriverWait(driver, 20)


# ── Helpers ───────────────────────────────────────────────────────────────────

def wait_for_content():
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".zw-contentpane")))
    except Exception:
        pass
    time.sleep(3)


def get_article_title():
    for sel in [".zln-title\\:article h1", ".zln-title\\:article", "h1"]:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        for el in els:
            t = el.text.strip()
            if t:
                return t
    return driver.title.split(" - ")[0].strip()


def get_article_body():
    panes = driver.find_elements(By.CSS_SELECTOR, ".zw-contentpane")
    if panes:
        texts = [p.text.strip() for p in panes if len(p.text.strip()) > 50]
        if texts:
            return max(texts, key=len)
    # Fallback
    body_text = driver.find_element(By.TAG_NAME, "body").text
    lines = [l.strip() for l in body_text.splitlines() if len(l.strip()) > 20]
    return "\n".join(lines)


def find_next_url():
    """Find the URL of the next article via the Zoho Learn footer nav button."""
    # Primary: Zoho Learn marks the next button with id='zln-next-icon'
    # or class containing 'zln-next:article'. We use the id as it's unique.
    els = driver.find_elements(By.ID, "zln-next-icon")
    if not els:
        # Fallback selector using attribute contains
        els = driver.find_elements(By.CSS_SELECTOR, "[class*='zln-next']")
    for el in els:
        cls  = el.get_attribute("class") or ""
        href = el.get_attribute("href") or ""
        # Skip if the button is disabled
        if "z-disabled" in cls:
            return None
        if href and "sri-ramakrishna-the-great-master" in href:
            # Make absolute if needed
            if href.startswith("/"):
                href = BASE_URL + href
            return href
    return None


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    chapters    = []
    visited     = set()
    current_url = START_URL

    print(f"Starting at: {current_url}\n")

    try:
        while current_url and current_url not in visited:
            visited.add(current_url)
            driver.get(current_url)
            wait_for_content()

            title = get_article_title()
            body  = get_article_body()

            print(f"[{len(chapters)+1:>3}] {title[:70]}")
            print(f"       {len(body):,} chars | {current_url.split('/')[-1]}")

            chapters.append((title, body))

            next_url = find_next_url()
            if next_url and next_url not in visited:
                current_url = next_url
            else:
                print("\nNo further 'Next' link — reached end of book.")
                break

    finally:
        driver.quit()

    if not chapters:
        print("ERROR: No chapters were scraped.")
        return

    lines = [
        "SRI RAMAKRISHNA THE GREAT MASTER",
        "By Swami Saradananda",
        "Translated by Swami Jagadananda",
        "Source: https://englishbooks.rkmm.org",
        "=" * 70, "",
    ]
    for title, body in chapters:
        lines += [f"\n{'=' * 70}", title.upper(), '=' * 70, "", body, ""]

    full_text = clean_text("\n".join(lines))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\nSaved {len(chapters)} chapters -> {OUTPUT_FILE}")
    print(f"Total: {len(full_text):,} characters")


if __name__ == "__main__":
    main()
