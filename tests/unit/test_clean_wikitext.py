import json
from pathlib import Path

import pytest

from wiki2shards.clean.wikitext import Policy, clean_wikitext, load_language_codes
from wiki2shards.config import CleanConfig

CASES = json.loads((Path(__file__).parents[1] / "golden/wikitext/cases.json").read_text())
_cfg = CleanConfig()
POLICY = Policy(_cfg.drop_sections, load_language_codes(_cfg.language_codes_file))


def test_golden_has_enough_cases() -> None:
    assert len(CASES) >= 25


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_wikitext_golden(case):
    assert clean_wikitext(case["wikitext"], POLICY)[0] == case["expected"]
