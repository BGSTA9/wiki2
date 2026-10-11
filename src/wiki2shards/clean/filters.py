"""Document filters: pure functions returning (name, keep).

The hard filter (empty after cleaning) drops a document in the clean stage. Soft filters only
record a verdict; the select stage decides which verdicts a policy applies.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable


def min_words(text: str, threshold: int) -> tuple[str, bool]:
    # Whitespace words: weak on CJK (no spaces), which is fine for a recorded soft verdict.
    return "min_words", len(text.split()) >= threshold


def max_symbol_pct(text: str, pct: int) -> tuple[str, bool]:
    """Symbols = non-whitespace chars outside Unicode L*, N*, M*. Integer math only."""
    n_symbols = sum(
        1 for ch in text if not ch.isspace() and unicodedata.category(ch)[0] not in "LNM"
    )
    return "max_symbol_pct", 100 * n_symbols <= pct * len(text)


def normalize_template_name(name: str) -> str:
    name = " ".join(name.replace("_", " ").split()).casefold()
    if name.startswith("template:"):
        name = name[len("template:") :].strip()
    return name


def disambiguation(template_names: Iterable[str], listed: Iterable[str]) -> tuple[str, bool]:
    wanted = {normalize_template_name(n) for n in listed}
    found = any(normalize_template_name(n) in wanted for n in template_names)
    return "disambiguation", not found


def list_page(title: str) -> tuple[str, bool]:
    return "list_page", not title.startswith("List of")
