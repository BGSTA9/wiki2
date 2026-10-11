"""Write configs/language_codes.txt: every Wikipedia language edition's code (interlanguage
link prefixes such as de:, fr:, simple:). One network request to the Wikimedia sitematrix API;
run it once and commit the result. Tests never call this.

Usage: uv run python scripts/fetch_language_codes.py [--out configs/language_codes.txt]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from wiki2shards.acquire.download import USER_AGENT

API = "https://meta.wikimedia.org/w/api.php?action=sitematrix&format=json&smsiteprop=code"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("configs/language_codes.txt"))
    args = ap.parse_args()
    req = urllib.request.Request(API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        matrix = json.load(resp)["sitematrix"]
    codes = set()
    for key, lang in matrix.items():
        if key in ("count", "specials") or not isinstance(lang, dict):
            continue
        if any(site.get("code") == "wiki" for site in lang.get("site", [])):
            codes.add(lang["code"].casefold())
    header = "# Wikipedia language-edition codes from the Wikimedia sitematrix API.\n"
    args.out.write_text(header + "".join(f"{c}\n" for c in sorted(codes)), encoding="utf-8")
    print(f"{len(codes)} codes -> {args.out}")


if __name__ == "__main__":
    main()
