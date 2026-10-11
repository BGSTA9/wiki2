import unicodedata

from hypothesis import given
from hypothesis import strategies as st

from wiki2shards.clean.normalize import normalize

# Characters that break naive normalizers: controls (BEL, NEL, DEL, CR), NBSP, combining marks,
# Unicode spaces, the line separator, a zero-width space, Hebrew, Hangul jamo and an emoji.
TRICKY = st.sampled_from(
    [
        *("e", "a", "x", " ", "\t", "\n", "\r", "\x07", "\x85", "\x7f"),
        "\N{NO-BREAK SPACE}",
        "\N{COMBINING ACUTE ACCENT}",
        "\N{COMBINING DIAERESIS}",
        "\N{COMBINING CEDILLA}",
        "\N{EN QUAD}",
        "\N{IDEOGRAPHIC SPACE}",
        "\N{LINE SEPARATOR}",
        "\N{ZERO WIDTH SPACE}",
        "\N{HEBREW LETTER ALEF}",
        "\N{HANGUL CHOSEONG KIYEOK}",
        "\N{HANGUL JUNGSEONG A}",
        "\N{GRINNING FACE}",
    ]
)
TEXT = st.one_of(st.text(), st.lists(TRICKY, max_size=40).map("".join))


@given(TEXT)
def test_normalize_idempotent(s):
    once = normalize(s)
    assert normalize(once) == once
    assert unicodedata.normalize("NFC", once) == once
    assert not any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in once)
    assert "\N{NO-BREAK SPACE}" not in once and "  " not in once and "\n\n\n" not in once
    assert all(line == line.strip() for line in once.split("\n"))


def test_spec_order_counterexample():
    """SPEC's original order (NFC first) is not idempotent on this input; ours is."""
    assert normalize("e\x07\N{COMBINING ACUTE ACCENT}") == "\N{LATIN SMALL LETTER E WITH ACUTE}"
