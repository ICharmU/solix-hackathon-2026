"""
AUTO-03: In-IDE Rapid Documentation & Context Resolver.
Fetches schema specifications and API documentation directly within the terminal/IDE,
preventing context switching to external browsers.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any


def search_local_references(query: str, search_dir: Path) -> List[Dict[str, Any]]:
    """Searches workspace markdown and source files for reference query, skipping virtualenvs and hidden dirs."""
    matches = []
    ignored_parts = {".pixi", ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv"}
    for p in search_dir.rglob("*"):
        if any(part in ignored_parts or part.startswith(".") for part in p.parts):
            continue
        if p.is_file() and p.suffix in [".md", ".py", ".json", ".txt"]:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                for line_num, line in enumerate(text.splitlines(), start=1):
                    if query.lower() in line.lower():
                        matches.append({
                            "file": str(p.relative_to(search_dir)),
                            "line": line_num,
                            "content": line.strip()
                        })
            except Exception:
                continue
    return matches


def main():
    parser = argparse.ArgumentParser(description="In-IDE Rapid Documentation Resolver (AUTO-03)")
    parser.add_argument("query", type=str, help="Search keyword or schema name")
    parser.add_argument("--dir", type=str, default=".", help="Workspace search directory")
    args = parser.parse_args()

    results = search_local_references(args.query, Path(args.dir))
    print(f"--- Search Results for '{args.query}' ({len(results)} matches) ---")
    for r in results[:10]:
        print(f"  • {r['file']}:{r['line']}  {r['content']}")


if __name__ == "__main__":
    main()
