"""
AUTO-02: Programmatic Web Table & Schema Validator.
Automates validation of web documentation tables against expected data models,
eliminating repeated cross-application context switching to Google Chrome.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import urllib.request
from html.parser import HTMLParser


class TableHTMLParser(HTMLParser):
    """Lightweight pure-python HTML parser extracting table headers and rows."""

    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_th = False
        self.in_td = False
        self.current_headers = []
        self.all_tables = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.current_headers = []
        elif tag == "th" and self.in_table:
            self.in_th = True
        elif tag == "td" and self.in_table:
            self.in_td = True

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
            if self.current_headers:
                self.all_tables.append(self.current_headers)
        elif tag == "th":
            self.in_th = False
        elif tag == "td":
            self.in_td = False

    def handle_data(self, data):
        if self.in_th and data.strip():
            self.current_headers.append(data.strip())


def validate_table_schema(html_content: str, expected_columns: List[str]) -> Tuple[bool, List[str], List[str]]:
    """
    Validates that extracted table headers contain expected columns.
    Returns (is_valid, matched_columns, missing_columns).
    """
    parser = TableHTMLParser()
    parser.feed(html_content)

    found_headers = []
    for table_cols in parser.all_tables:
        found_headers.extend(table_cols)

    found_set = set(h.lower() for h in found_headers)
    expected_set = set(c.lower() for c in expected_columns)

    missing = [c for c in expected_columns if c.lower() not in found_set]
    matched = [c for c in expected_columns if c.lower() in found_set]

    is_valid = len(missing) == 0
    return is_valid, matched, missing


def main():
    parser = argparse.ArgumentParser(description="Automated Web Table & Schema Validator (AUTO-02)")
    parser.add_argument("--url", type=str, help="URL to inspect")
    parser.add_argument("--file", type=str, help="Local HTML file to inspect")
    parser.add_argument("--expected", nargs="+", default=["Timestamp", "Application", "Status"], help="Expected column names")
    args = parser.parse_args()

    content = ""
    if args.file and Path(args.file).exists():
        content = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    elif args.url:
        req = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
    else:
        print("[ERROR] Please provide --url or --file")
        sys.exit(1)

    is_valid, matched, missing = validate_table_schema(content, args.expected)
    print(f"Validation Result: {'[PASS]' if is_valid else '[FAIL]'}")
    print(f"Matched Columns:  {matched}")
    print(f"Missing Columns:  {missing}")

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
