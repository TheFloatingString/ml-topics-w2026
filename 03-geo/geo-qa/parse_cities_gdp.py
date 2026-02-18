#!/usr/bin/env python3
"""Parse Wikipedia cities by GDP HTML and extract data to CSV using BeautifulSoup."""

import csv
from pathlib import Path
from bs4 import BeautifulSoup


def clean_text(text):
    """Clean up text by removing extra whitespace."""
    if not text:
        return ""
    return " ".join(text.split())


def extract_city_name(td):
    """Extract city name from table cell."""
    # Get all text from the cell
    text = td.get_text(strip=True)
    return clean_text(text)


def extract_country_name(td):
    """Extract country name from table cell."""
    # Find all links in the cell
    links = td.find_all("a")

    # Filter out EU, ASEAN and get the actual country
    for link in reversed(links):
        text = link.get_text(strip=True)
        if text and text not in ["European Union", "ASEAN"]:
            return text

    # Fallback to all text
    text = td.get_text(strip=True)
    return clean_text(text)


def extract_gdp(td):
    """Extract GDP value from table cell."""
    # Look for data-sort-value attribute
    sort_value = td.get("data-sort-value", "")
    if sort_value:
        # Format is like: 7001323440000000000♠32.344
        if "♠" in sort_value:
            parts = sort_value.split("♠")
            if len(parts) > 1:
                return parts[1].strip()
        return sort_value.strip()

    # Fallback: get text and extract number
    text = td.get_text(strip=True)
    # Extract number before parentheses
    import re

    match = re.search(r"([\d.,]+)", text)
    if match:
        return match.group(1)
    return ""


def extract_population(td):
    """Extract population value from table cell."""
    text = td.get_text(strip=True)
    # Extract number (with commas)
    import re

    match = re.search(r"([\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    return ""


def parse_table(html_content):
    """Parse the HTML table and extract city data."""
    cities = []

    soup = BeautifulSoup(html_content, "html.parser")

    # Find the table with class "static-row-numbers sortable wikitable"
    table = soup.find("table", {"class": "static-row-numbers sortable wikitable"})

    if not table:
        print("Table not found!")
        return cities

    print("Found table, parsing rows...")

    # Find all table rows
    rows = table.find_all("tr")

    print(f"Found {len(rows)} rows")

    for i, row in enumerate(rows):
        # Skip header rows
        if "static-row-header" in row.get("class", []) or row.find("th"):
            continue

        # Find all table cells
        cells = row.find_all("td")

        if len(cells) >= 4:
            city_name = extract_city_name(cells[0])
            country = extract_country_name(cells[1])
            gdp = extract_gdp(cells[2])
            population = extract_population(cells[3])

            # Skip if we couldn't extract basic info
            if not city_name or not gdp:
                continue

            # Calculate GDP per capita if we have both values
            gdp_per_capita = ""
            if gdp and population:
                try:
                    gdp_val = float(gdp.replace(",", ""))
                    pop_val = float(population.replace(",", ""))
                    if pop_val > 0:
                        gdp_per_capita = round((gdp_val * 1000000000) / pop_val, 2)
                except (ValueError, ZeroDivisionError):
                    pass

            cities.append(
                {
                    "city": city_name,
                    "country": country,
                    "gdp_billion_usd": gdp,
                    "population": population,
                    "gdp_per_capita": gdp_per_capita,
                }
            )

    return cities


def main():
    # Read the HTML file
    html_path = Path(
        r"C:\Users\laure\.local\share\opencode\tool-output\tool_c721735020015EljEE5fBjh3yx"
    )
    output_path = Path(
        r"C:\Users\laure\Projects\ml-topics-w2026\03-geo\geo-qa\cities_gdp.csv"
    )

    print(f"Reading HTML file: {html_path}")
    html_content = html_path.read_text(encoding="utf-8")

    print("Parsing table...")
    cities = parse_table(html_content)

    print(f"Extracted {len(cities)} cities")

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "city",
                "country",
                "gdp_billion_usd",
                "population",
                "gdp_per_capita",
            ],
        )
        writer.writeheader()
        writer.writerows(cities)

    print(f"CSV file created: {output_path}")

    # Print first few entries as sample
    print("\nSample data (first 10 cities):")
    for city in cities[:10]:
        print(
            f"  {city['city']}, {city['country']}: GDP=${city['gdp_billion_usd']}B, Pop={city['population']}, GDP/capita=${city['gdp_per_capita']}"
        )


if __name__ == "__main__":
    main()
