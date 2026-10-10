"""Write a deterministic multistream fixture dump (SPEC 1.7).

Layout mirrors dumps.wikimedia.org, so the download stage can use it as a local mirror:
    <out>/<wiki>/<date>/<wiki>-<date>-pages-articles-multistream.xml.bz2
    <out>/<wiki>/<date>/<wiki>-<date>-pages-articles-multistream-index.txt.bz2
    <out>/<wiki>/<date>/<wiki>-<date>-sha1sums.txt
    <out>/<wiki>/<date>/fixture_expected.json   (known answers; not part of the dump)

No PRNG anywhere: every "random" choice is derived from sha256, so the bytes are identical on every
machine and Python version. 100 pages per bz2 stream, like the real dumps.

Usage: python scripts/make_fixture_dump.py --out data/fixture_dump [--wiki fixturewiki]
       [--date 20260901] [--variant 1]
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

PAGES_PER_STREAM = 100
BASE_DATE = "20260901"
WORDS = [
    "river",
    "valley",
    "stone",
    "harbor",
    "north",
    "light",
    "garden",
    "market",
    "winter",
    "summer",
    "bridge",
    "tower",
    "field",
    "ancient",
    "modern",
    "history",
    "people",
    "language",
    "music",
    "science",
    "island",
    "mountain",
    "forest",
    "city",
    "village",
    "road",
    "council",
    "school",
    "library",
    "museum",
    "theatre",
    "church",
    "castle",
    "railway",
    "station",
    "canal",
    "lake",
    "coast",
    "province",
    "region",
    "empire",
    "kingdom",
    "republic",
    "treaty",
    "battle",
    "army",
    "navy",
    "trade",
    "industry",
    "farm",
    "mill",
    "copper",
    "iron",
    "silver",
    "gold",
    "coal",
    "salt",
    "wool",
    "cotton",
    "paper",
    "glass",
    "bronze",
    "timber",
    "grain",
    "wine",
    "bread",
    "poet",
    "painter",
    "scholar",
    "merchant",
    "engineer",
    "sailor",
    "farmer",
    "builder",
    "teacher",
    "judge",
    "mayor",
    "bishop",
    "early",
    "late",
    "great",
    "small",
    "long",
    "short",
    "high",
    "low",
    "old",
    "new",
    "first",
    "second",
    "third",
    "final",
    "central",
    "eastern",
    "western",
    "southern",
    "northern",
    "famous",
    "common",
    "rare",
    "local",
    "national",
    "public",
    "private",
    "royal",
    "built",
    "founded",
    "named",
    "moved",
    "opened",
    "closed",
    "grew",
    "became",
    "remained",
    "served",
    "described",
    "recorded",
    "the",
    "of",
    "and",
    "in",
    "to",
    "was",
    "is",
    "for",
    "on",
    "by",
    "with",
    "as",
    "at",
    "from",
    "that",
    "which",
    "its",
    "their",
    "during",
    "after",
]

COUNTS = {  # designed categories for variant 1; "article" fills the rest up to 300 pages
    "redirect": 20,
    "talk": 10,
    "user": 10,
    "non_wikitext": 3,
    "empty": 5,
    "short": 10,
    "disambiguation": 6,
    "duplicate": 8,
    "long": 2,
    "cjk": 1,
    "rtl": 1,
    "emoji": 1,
    "eos_literal": 1,
    "symbols": 1,
    "list_page": 2,
}
TOTAL_PAGES = 300


def _h(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def words(seed: str, n: int) -> list[str]:
    return [WORDS[_h(seed, i) % len(WORDS)] for i in range(n)]


def prose(seed: str, n_words: int) -> str:
    """Sentences of 8-15 words, grouped into paragraphs of about five sentences."""
    out, sentence, paragraph = [], [], []
    for i, w in enumerate(words(seed, n_words)):
        sentence.append(w)
        if len(sentence) >= 8 + _h(seed, "len", i) % 8:
            paragraph.append(" ".join(sentence).capitalize() + ".")
            sentence = []
            if len(paragraph) == 5:
                out.append(" ".join(paragraph))
                paragraph = []
    if sentence:
        paragraph.append(" ".join(sentence).capitalize() + ".")
    if paragraph:
        out.append(" ".join(paragraph))
    return "\n\n".join(out)


def article_wikitext(seed: str, k: int) -> str:
    """A long article exercising every construct the clean policy must handle."""
    p1, p2, p3 = prose(seed + "a", 40), prose(seed + "b", 40), prose(seed + "c", 30)
    extras = [
        "{{Infobox settlement|name=Example|population=1000|image=Example.jpg}}\n",
        "{{Use dmy dates|date=August 2026}}\n",
        "",
    ][k % 3]
    return (
        f"{extras}'''{seed.title()}''' is a [[Paris|place]] in the [[Ruritania]] region."
        f"<ref>{{{{cite web|url=http://example.com/{k}|title=Source {k}}}}}</ref>\n\n"
        f'{p1}<ref name="n{k}">Smith, p. {k}.</ref>\n\n'
        f"[[File:Example {k}.jpg|thumb|A caption about [[stone]] bridges]]\n"
        f"== History ==\n{p2} See also &amp; compare <math>x^{k % 9}</math> values.\n"
        f"<!-- editor note {k} -->\n"
        '{| class="wikitable"\n|-\n! Year !! Value\n|-\n| 1900 || 12\n|}\n'
        f"=== Later period ===\n{p3} Visit [http://example.org/{k} the archive] or http://example.net.\n"
        "== References ==\n<references />\n"
        f"== External links ==\n* [http://example.com/{k} Official site]\n"
        f"[[Category:Fixture articles]]\n[[de:Beispiel {k}]]\n[[fr:Exemple {k}]]\n"
    )


@dataclass
class Page:
    key: str
    category: str
    title: str
    ns: int
    wikitext: str
    model: str = "wikitext"
    redirect: str | None = None


def build_pages(wiki: str, variant: int) -> list[Page]:
    pages: list[Page] = []
    cats = {**COUNTS, "article": TOTAL_PAGES - sum(COUNTS.values())}
    for cat in sorted(cats):
        for k in range(cats[cat]):
            key = f"{cat}-{k}"
            seed = f"{wiki}-{key}"
            title = " ".join(words(seed + "t", 2)).title() + f" {cat.title()} {k}"
            text, ns, model, redirect = article_wikitext(seed, k), 0, "wikitext", None
            if cat == "redirect":
                redirect = f"Target {k}"
                text = f"#REDIRECT [[{redirect}]]"
            elif cat == "talk":
                ns, title = 1, f"Talk:{title}"
            elif cat == "user":
                ns, title = 2, f"User:{title}"
            elif cat == "non_wikitext":
                model, text = "json", json.dumps({"k": k, "words": words(seed, 5)})
            elif cat == "empty":
                text = (
                    "{{Infobox thing|a=1}}\n<ref>Only a reference.</ref>\n<!-- nothing -->\n"
                    "{|\n| a cell\n|}\n[[Category:Empty pages]]\n[[File:X.png|thumb|caption]]\n"
                )
            elif cat == "short":
                text = prose(seed, 12)
            elif cat == "disambiguation":
                tmpl = [
                    "disambiguation",
                    "Disambig",
                    "dab",
                    "hndis|name=X",
                    "geodis",
                    "Template:Disambiguation",
                ][k % 6]
                items = "\n".join(
                    f"* [[{w.title()} ({k})]], {prose(seed + w, 12)}" for w in words(seed + "i", 5)
                )
                text = f"'''{title}''' may refer to:\n{items}\n\n{{{{{tmpl}}}}}\n"
            elif cat == "duplicate":
                text = article_wikitext(f"{wiki}-dup-{k // 2}", k // 2)  # pairs share wikitext
            elif cat == "long":
                text = prose(seed, 3000)
            elif cat == "cjk":
                text = "".join(f"北京是第{i}个城市的首都。上海位于长江口。" for i in range(40))
            elif cat == "rtl":
                text = " ".join(["שלום", "עולם", "مرحبا", "بالعالم", "ספר", "كتاب"] * 12)
            elif cat == "emoji":
                ws = words(seed, 80)
                text = " ".join(w + (" 🙂" if i % 10 == 0 else "") for i, w in enumerate(ws))
            elif cat == "eos_literal":
                text = prose(seed, 60) + " The marker <|endoftext|> must stay plain text."
            elif cat == "symbols":
                text = " ".join(f"{w} @@@ ### %%% ***" for w in words(seed, 60))
            elif cat == "list_page":
                title = f"List of {words(seed, 1)[0]} things {k}"
            pages.append(Page(key, cat, title, ns, text, model, redirect))

    # Deterministic shuffle so every stream mixes categories.
    pages.sort(key=lambda p: _h(wiki, "order", p.key))
    if variant == 2:  # a later dump: some edits, some deletions, some new pages
        removed = {p.key for p in pages if p.category == "article"}
        removed = set(sorted(removed, key=lambda key: _h(wiki, "rm", key))[:5])
        pages = [p for p in pages if p.key not in removed]
        for p in pages:
            if p.category == "article" and _h(wiki, "edit", p.key) % 5 == 0:
                p.wikitext += "\n\nAn edit made for the later dump.\n"
        pages += [
            Page(f"new-{k}", "article", f"New Page {k}", 0, article_wikitext(f"{wiki}-new-{k}", k))
            for k in range(5)
        ]
    return pages


def page_ids(wiki: str, pages: list[Page]) -> dict[str, int]:
    """Stable page ids: ids come from the variant-1 page set, so a page keeps its id in later
    dumps even when other pages are deleted; pages new in a later dump get ids after all of them."""
    base = [p.key for p in build_pages(wiki, 1)]
    ids = {key: 10 + 3 * i for i, key in enumerate(base)}
    new = sorted(p.key for p in pages if p.key not in ids)
    ids.update({key: 10 + 3 * (len(base) + j) for j, key in enumerate(new)})
    return {p.key: ids[p.key] for p in pages}


def page_xml(p: Page, page_id: int, rev_id: int) -> str:
    text = p.wikitext
    redirect = f"    <redirect title={quoteattr(p.redirect)} />\n" if p.redirect else ""
    sha1 = hashlib.sha1(text.encode()).hexdigest()
    return (
        "  <page>\n"
        f"    <title>{escape(p.title)}</title>\n"
        f"    <ns>{p.ns}</ns>\n"
        f"    <id>{page_id}</id>\n"
        f"{redirect}"
        "    <revision>\n"
        f"      <id>{rev_id}</id>\n"
        f"      <parentid>{rev_id - 1}</parentid>\n"
        "      <timestamp>2026-08-01T00:00:00Z</timestamp>\n"
        "      <contributor>\n        <username>Fixture</username>\n        <id>7</id>\n"
        "      </contributor>\n"
        f"      <model>{p.model}</model>\n"
        f"      <format>{'text/x-wiki' if p.model == 'wikitext' else 'application/json'}</format>\n"
        f'      <text bytes="{len(text.encode())}" xml:space="preserve">{escape(text)}</text>\n'
        f"      <sha1>{sha1}</sha1>\n"
        "    </revision>\n"
        "  </page>\n"
    )


def header_xml(wiki: str) -> str:
    return (
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.11/ '
        'http://www.mediawiki.org/xml/export-0.11.xsd" version="0.11" xml:lang="en">\n'
        "  <siteinfo>\n"
        "    <sitename>Fixturepedia</sitename>\n"
        f"    <dbname>{wiki}</dbname>\n"
        "    <base>https://fixture.example/wiki/Main_Page</base>\n"
        "    <generator>MediaWiki 1.45.0</generator>\n"
        "    <case>first-letter</case>\n"
        "    <namespaces>\n"
        '      <namespace key="-2" case="first-letter">Media</namespace>\n'
        '      <namespace key="0" case="first-letter" />\n'
        '      <namespace key="1" case="first-letter">Talk</namespace>\n'
        '      <namespace key="2" case="first-letter">User</namespace>\n'
        '      <namespace key="6" case="first-letter">File</namespace>\n'
        '      <namespace key="14" case="first-letter">Category</namespace>\n'
        "    </namespaces>\n"
        "  </siteinfo>\n"
    )


def expected_answers(pages: list[Page], ids: dict[str, int]) -> dict[str, object]:
    by_cat: dict[str, list[int]] = {}
    for p in pages:
        by_cat.setdefault(p.category, []).append(ids[p.key])
    n = {cat: len(v) for cat, v in by_cat.items()}
    kept = len(pages) - n["redirect"] - n["talk"] - n["user"] - n["non_wikitext"]
    cleaned = kept - n["empty"]
    fail_min_words = n["short"] + n["cjk"]  # CJK has no spaces, so it counts as few words
    after_filters = cleaned - fail_min_words - n["disambiguation"]
    return {
        "page_ids": {cat: sorted(v) for cat, v in sorted(by_cat.items())},
        "extract": {
            "pages": len(pages),
            "kept": kept,
            "skipped_redirect": n["redirect"],
            "skipped_namespace": n["talk"] + n["user"],
            "skipped_model": n["non_wikitext"],
        },
        "clean": {"docs": cleaned, "dropped_empty": n["empty"]},
        "verdicts_false": {
            "min_words": fail_min_words,
            "disambiguation": n["disambiguation"],
            "max_symbol_pct": n["symbols"],
            "list_page": n["list_page"],
        },
        "select": {
            "after_filters": after_filters,
            "dedup_removed": n["duplicate"] // 2,
            "selected": after_filters - n["duplicate"] // 2,
        },
    }


def make_dump(out: Path, wiki: str, date: str, variant: int) -> Path:
    pages = build_pages(wiki, variant)
    ids = page_ids(wiki, pages)
    pages.sort(key=lambda p: ids[p.key])  # dumps are in page_id order
    rev_base = 1000 if variant == 1 else 5000

    streams = [bz2.compress(header_xml(wiki).encode())]
    index_lines: list[str] = []
    offset = len(streams[0])
    for s in range(0, len(pages), PAGES_PER_STREAM):
        chunk = pages[s : s + PAGES_PER_STREAM]
        xml = "".join(page_xml(p, ids[p.key], rev_base + ids[p.key]) for p in chunk)
        index_lines += [f"{offset}:{ids[p.key]}:{p.title}\n" for p in chunk]
        streams.append(bz2.compress(xml.encode()))
        offset += len(streams[-1])
    streams.append(bz2.compress(b"</mediawiki>\n"))

    d = out / wiki / date
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{wiki}-{date}"
    xml_name = f"{stem}-pages-articles-multistream.xml.bz2"
    idx_name = f"{stem}-pages-articles-multistream-index.txt.bz2"
    (d / xml_name).write_bytes(b"".join(streams))
    (d / idx_name).write_bytes(bz2.compress("".join(index_lines).encode()))
    sums = "".join(
        f"{hashlib.sha1((d / n).read_bytes()).hexdigest()}  {n}\n"
        for n in sorted([xml_name, idx_name])
    )
    (d / f"{stem}-sha1sums.txt").write_text(sums)
    answers = expected_answers(pages, ids)
    (d / "fixture_expected.json").write_text(json.dumps(answers, indent=2, sort_keys=True) + "\n")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wiki", default="fixturewiki")
    ap.add_argument("--date", default=BASE_DATE)
    ap.add_argument("--variant", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()
    print(make_dump(args.out, args.wiki, args.date, args.variant))


if __name__ == "__main__":
    main()
