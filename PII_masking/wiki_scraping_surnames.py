import requests
from bs4 import BeautifulSoup
import time
import re

def get_wiktionary_page(url: str) -> BeautifulSoup:
    """Fetch and parse a Wiktionary page, returning a BeautifulSoup object."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_surnames(soup: BeautifulSoup) -> set:
    """Extract surnames from a Wiktionary appendix page."""
    surnames = set()
    if not soup:
        return surnames

    # Target tables or lists containing surnames
    tables = soup.find_all("table", class_="wikitable")
    if not tables:
        lists = soup.find_all("ul")
    else:
        lists = []

    # Process tables
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:  # Skip header
            cells = row.find_all("td")
            if cells:
                surname = cells[0].get_text(strip=True)
                if surname:
                    surname = re.sub(r"\[\d+\]|\([^)]*\)|[*]+", "", surname).strip()
                    if surname:
                        surnames.add(surname.lower())

    # Process lists
    for ul in lists:
        items = ul.find_all("li")
        for item in items:
            surname = item.get_text(strip=True)
            surname = re.sub(r"\[\d+\]|\([^)]*\)|[*]+", "", surname).strip()
            if surname:
                surnames.add(surname.lower())

    return surnames

def save_surnames(surnames: set, output_file: str) -> None:
    """Save surnames to a text file, one per line."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for surname in sorted(surnames):
                f.write(f"{surname}\n")
        print(f"Saved {len(surnames)} surnames to {output_file}")
    except Exception as e:
        print(f"Error saving surnames to {output_file}: {e}")

def main():
    # Define Wiktionary appendix URLs (1-6000)
    base_url = "https://en.wiktionary.org/wiki/Appendix:Filipino_surnames_"
    ranges = [
        "(1-1000)", "(1001-2000)", "(2001-3000)", 
        "(3001-4000)", "(4001-5000)", "(5001-6000)"
    ]
    urls = [f"{base_url}{r}" for r in ranges]
    output_file = "filipino_surnames.txt"
    all_surnames = set()

    for url in urls:
        print(f"Scraping {url}...")
        soup = get_wiktionary_page(url)
        surnames = extract_surnames(soup)
        all_surnames.update(surnames)
        print(f"Extracted {len(surnames)} surnames from {url}")
        time.sleep(1)  # Avoid rate limiting

    # Save to file
    save_surnames(all_surnames, output_file)

if __name__ == "__main__":
    main()