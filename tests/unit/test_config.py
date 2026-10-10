import pytest
from helpers.build import fixture_config

from wiki2shards.config import BuildConfig, ConfigError, load_build_config, load_mix_config


def test_fixture_config_loads():
    cfg = load_build_config("configs/fixture.yaml")
    assert cfg.pack.row_len == 65 and cfg.pack.effective_last_row == "drop"
    assert cfg.clean.soft_filters.enabled() == [
        "disambiguation",
        "list_page",
        "max_symbol_pct",
        "min_words",
    ]


@pytest.mark.parametrize(
    "patch",
    [
        {"source": {"wiki": "x", "date": 20260901}},  # unquoted YAML date -> int
        {"pack": {"seq_len": 8, "typo": 1}},  # unknown key
        {"pack": {"mode": "bfd", "seq_len": 8, "last_row": "drop"}},
        {"select": {"apply": ["list_page"]}},  # filter clean does not compute
        {"select": {"val_per_10k": 10_001}},
    ],
)
def test_config_rejects(patch):
    base = {
        "name": "x",
        "source": {"wiki": "x", "date": "20260901"},
        "workdir": "w",
        "output": "o",
        "tokenizer": "tiktoken:gpt2",
        "pack": {"seq_len": 8},
        "clean": {"soft_filters": {"list_page": False}},
    }
    base.update(patch)
    with pytest.raises(ValueError):
        BuildConfig.model_validate(base)


def test_config_hash_ignores_execution_fields(tmp_path, dumps):
    gpt2 = {"tokenizer": "tiktoken:gpt2"}  # hashing the config never loads the tokenizer
    a = fixture_config(tmp_path / "a", dumps, **gpt2)
    b = fixture_config(tmp_path / "b", dumps, workers=7, **gpt2)
    c = fixture_config(tmp_path / "c", dumps, pack__seq_len=32, **gpt2)
    assert a.config_sha256() == b.config_sha256()  # workers, workdir, output, mirror excluded
    assert a.config_sha256() != c.config_sha256()


def test_mix_config_errors(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("seed: 1\nsources:\n  - {name: a, path: x, weight: 0}\n")
    with pytest.raises(ConfigError):
        load_mix_config(p)
