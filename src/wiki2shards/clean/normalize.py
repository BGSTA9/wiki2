"""Text normalization. Must be idempotent: normalize(normalize(x)) == normalize(x).

Order differs from SPEC 1.3 (PLAN A2): controls are removed and U+00A0 mapped *before* NFC.
SPEC's order (NFC first) is not idempotent: "e\\x07\\u0301" survives NFC unchanged because BEL
blocks composition, BEL is then removed, and a second pass composes "e\\u0301" to "é".
"""

from __future__ import annotations

import re
import unicodedata

# Unicode category Cc minus \t and \n: C0 controls, DEL and C1 controls. NBSP becomes a space.
_TABLE: dict[int, str | None] = {
    cp: None for cp in [*range(0x00, 0x20), *range(0x7F, 0xA0)] if cp not in (0x09, 0x0A)
}
_TABLE[0xA0] = " "
_SPACES = re.compile(r"[ \t]+")
_BLANK_RUNS = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = text.translate(_TABLE)
    text = unicodedata.normalize("NFC", text)
    text = _SPACES.sub(" ", text)
    # Split on "\n" only: str.splitlines() also splits on \x0b, \x0c, \x1c-\x1e, \x85, U+2028/9.
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()