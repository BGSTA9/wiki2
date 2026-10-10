"""Shared test setup: Hypothesis profiles, a network guard, and session-scoped fixture builds."""

from __future__ import annotations

import atexit
import os
import shutil
import socket
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings
from hypothesis.configuration import set_hypothesis_home_dir

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)  # configs use CWD-relative paths (tokenizer, language codes)

# Hypothesis keeps caches (constants, unicode data) in ./.hypothesis even with database=None.
# Tests write only under temporary directories, so point its home at one for this session.
_HYPOTHESIS_HOME = tempfile.mkdtemp(prefix="w2s-hypothesis-")
atexit.register(shutil.rmtree, _HYPOTHESIS_HOME, True)
set_hypothesis_home_dir(_HYPOTHESIS_HOME)

# "ci" is deterministic and writes no example database (tests write only under tmp_path).
settings.register_profile(
    "ci",
    max_examples=60,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("explore", max_examples=500, deadline=None, database=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never touch the network: refuse any internet socket connection."""
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: object) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(f"network access attempted in a test: {address!r}")
        real_connect(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(scope="session")
def dumps(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture dumps: fixturewiki at two dates, plus a second wiki for two-source mixtures."""
    from helpers.build import make_dumps

    return make_dumps(tmp_path_factory.mktemp("dumps"))


@pytest.fixture(scope="session")
def fixture_build(dumps: Path, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    from helpers.build import build_fixture

    return build_fixture(tmp_path_factory.mktemp("build_a"), dumps)


@pytest.fixture(scope="session")
def fixture_build_b(dumps: Path, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    from helpers.build import build_fixture

    return build_fixture(tmp_path_factory.mktemp("build_b"), dumps, wiki="fixture2wiki")
