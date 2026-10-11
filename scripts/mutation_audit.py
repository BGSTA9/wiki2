# ruff: noqa: E501  (patch targets must match the source text exactly)
"""Mutation audit (SPEC 1.7): each mutation is a real silent bug. Apply it in an isolated copy, run
the tests that must catch it, and report. A mutation that leaves its tests green is a defect in
the suite: fix the suite, never the mutation list.

Usage: uv run python scripts/mutation_audit.py [--only NAME] [--jobs N]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = "src/wiki2shards"


@dataclass(frozen=True)
class Mutation:
    name: str
    patches: tuple[tuple[str, str, str], ...]  # (file, old, new): old must occur exactly once
    must_fail: tuple[str, ...]  # named tests (pytest node ids under tests/)


LDF = """def simulate_edf(weights: Sequence[int], n_slots: int) -> list[int]:
    D = sum(weights)
    credit = [0] * len(weights)
    out = []
    for _ in range(n_slots):  # MUTANT: largest-deficit-first (smooth weighted round-robin)
        credit = [c + w for c, w in zip(credit, weights)]
        s = max(range(len(weights)), key=lambda i: (credit[i], -i))
        credit[s] -= D
        out.append(s)
    return out


def _unused_edf(weights: Sequence[int], n_slots: int) -> list[int]:"""

IID = """def simulate_edf(weights: Sequence[int], n_slots: int) -> list[int]:
    rng = np.random.Generator(np.random.PCG64(0))  # MUTANT: i.i.d. weighted sampling
    p = np.asarray(weights, dtype=np.float64) / sum(weights)
    return [int(x) for x in rng.choice(len(weights), size=n_slots, p=p)]


def _unused_edf(weights: Sequence[int], n_slots: int) -> list[int]:"""

NO_LAG_CHECK = (f"{SRC}/loader/mixture.py", "        sched.check_lag()\n", "")

COMPLETION_ORDER = """        order = [job.unit for job, _ in results]  # MUTANT: outputs named by completion order
        parts = sorted(u for u in order if u.startswith("part-"))
        rename = dict(zip([u for u in order if u.startswith("part-")], parts, strict=True))
        for job, files in results:
            for f in files:
                os.replace(state.out_dir / f, state.out_dir / (f + ".swap"))
        for job, files in results:
            new_unit = rename.get(job.unit, job.unit)
            new_files = [f.replace(job.unit, new_unit) for f in files]
            for f, g in zip(files, new_files, strict=True):
                os.replace(state.out_dir / (f + ".swap"), state.out_dir / g)
            commit(Job(new_unit, job.fn, job.args), new_files)
        for job, files in []:"""

MUTATIONS = [
    Mutation(
        "consumed_on_produce",
        (
            (
                f"{SRC}/loader/loader.py",
                "        self._consumed += self.G  # consumed, not produced\n",
                "        if self.prefetch <= 0:\n            self._consumed += self.G\n",
            ),
            (
                f"{SRC}/loader/loader.py",
                "                    q.put(item, timeout=0.05)\n",
                "                    q.put(item, timeout=0.05)\n                    self._consumed += self.G\n",
            ),
        ),
        (
            "integration/test_loader_prefetch.py::test_prefetch_state_is_consumed_not_produced",
            "integration/test_resume.py::test_sigkill_resume_bit_identical",
        ),
    ),
    Mutation(
        "edf_to_largest_deficit_first",
        (
            (
                f"{SRC}/loader/mixture.py",
                "def simulate_edf(weights: Sequence[int], n_slots: int) -> list[int]:",
                LDF,
            ),
            NO_LAG_CHECK,
        ),
        ("property/test_loader_mixture.py::test_mixture_lag_exhaustive",),
    ),
    Mutation(
        "edf_to_iid",
        (
            (
                f"{SRC}/loader/mixture.py",
                "def simulate_edf(weights: Sequence[int], n_slots: int) -> list[int]:",
                IID,
            ),
            NO_LAG_CHECK,
        ),
        (
            "property/test_loader_mixture.py::test_mixture_lag_exhaustive",
            "integration/test_mixture_e2e.py::test_mixture_end_to_end",
        ),
    ),
    Mutation(
        "doc_uid_builtin_hash",
        (
            (
                f"{SRC}/io/hashing.py",
                '    digest = hashlib.sha256(f"{wiki}:{page_id}".encode()).digest()\n'
                '    return int.from_bytes(digest[:8], "little")',
                '    return hash(f"{wiki}:{page_id}") & ((1 << 64) - 1)  # MUTANT',
            ),
        ),
        (
            "integration/test_build.py::test_hashseed_invariance",
            "unit/test_select_split.py::test_split_stable",
        ),
    ),
    Mutation(
        "pool_results_in_completion_order",
        (
            (
                f"{SRC}/pipeline/runner.py",
                "        for job, files in results:  # completion order: harmless, since files are named"
                " by unit",
                COMPLETION_ORDER,
            ),
        ),
        ("integration/test_build.py::test_worker_invariance",),
    ),
    Mutation(
        "write_final_name_directly",
        ((f"{SRC}/io/atomic.py", "    tmp = tmp_name(final)\n", "    tmp = final  # MUTANT\n"),),
        ("integration/test_build_crash.py::test_build_crash_resume",),
    ),
    Mutation(
        "cast_without_range_check",
        (
            (
                f"{SRC}/io/binfmt.py",
                "    if ids.size:\n        lo, hi",
                "    if False:\n        lo, hi",
            ),
        ),
        ("unit/test_io_binfmt.py::test_dtype_guard",),
    ),
    Mutation(
        "tokenizer_parses_special_tokens",
        (
            (
                f"{SRC}/tokenize/hf_tok.py",
                "        tok.encode_special_tokens = True\n",
                "        tok.encode_special_tokens = False  # MUTANT\n",
            ),
            (
                f"{SRC}/tokenize/tiktoken_tok.py",
                "self._enc.encode_ordinary_batch(list(texts))",
                'self._enc.encode_batch(list(texts), allowed_special="all")',
            ),
        ),
        ("unit/test_tokenize_special.py::test_special_token_injection",),
    ),
    Mutation(
        "modulo_instead_of_cycle_walking",
        (
            (
                f"{SRC}/loader/permutation.py",
                "        while x >= self.n:\n            x = self._encrypt(x)\n        return x",
                "        return x % self.n  # MUTANT",
            ),
            (
                f"{SRC}/loader/permutation.py",
                "        out = x >= n\n        while np.any(out):\n            x[out] = self._encrypt_vec(x[out])\n"
                "            out = x >= n\n        return x",
                "        return x % n  # MUTANT",
            ),
        ),
        ("unit/test_loader_permutation.py::test_feistel_bijection",),
    ),
    Mutation(
        "timestamp_in_side_file",
        (
            (
                f"{SRC}/pack/writer.py",
                '    return {"selection": selection_stats, "splits": splits}',
                "    import time  # MUTANT\n\n"
                '    return {"selection": selection_stats, "splits": splits,'
                ' "generated_at": int(time.time())}',
            ),
        ),
        ("integration/test_build.py::test_build_twice_identical",),
    ),
]


def run(m: Mutation) -> tuple[Mutation, set[str], str]:
    with tempfile.TemporaryDirectory(prefix=f"w2s-mut-{m.name}-") as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(
            REPO,
            copy,
            ignore=shutil.ignore_patterns(
                ".venv", "data", ".git", "__pycache__", ".mypy_cache", ".ruff_cache"
            ),
        )
        for rel, old, new in m.patches:
            p = copy / rel
            text = p.read_text()
            if text.count(old) != 1:
                return m, set(), f"patch target not found exactly once in {rel}"
            p.write_text(text.replace(old, new))
        env = {**os.environ, "PYTHONPATH": str(copy / "src")}  # shadow the editable install
        junit = copy / "junit.xml"
        tests = sorted({f"tests/{t}" for t in m.must_fail})
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                f"--junitxml={junit}",
                *tests,
            ],
            cwd=copy,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        failed = set()
        for case in ET.parse(junit).getroot().iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failed.add(case.attrib["name"].split("[")[0])
        return m, failed, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    todo = [m for m in MUTATIONS if args.only in (None, m.name)]
    clean = True
    with ThreadPoolExecutor(args.jobs) as pool:
        for m, failed, problem in pool.map(run, todo):
            wanted = {t.split("::")[1] for t in m.must_fail}
            missed = sorted(wanted - failed)
            ok = not missed and not problem
            clean &= ok
            status = "CAUGHT" if ok else "MISSED"
            print(
                f"{status:7s} {m.name:34s} failed: {', '.join(sorted(failed)) or '-'}"
                + (f"  | NOT caught by: {', '.join(missed)}" if missed else "")
                + (f"  | {problem}" if problem else "")
            )
    print("mutation audit:", "clean" if clean else "DEFECTS IN THE SUITE")
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
