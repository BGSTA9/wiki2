# wiki2shards

**wiki2shards** is a CPU-only Python package designed to turn Wikipedia database dumps into training-ready data shards for machine learning models (like GPT-2).

Our core philosophy is: **Correctness comes first, throughput second, features third.**

This project exists to prevent "silent bugs" in pre-training data pipelines—such as data repeating after a crash, dataset mixtures drifting, or two identically configured builds yielding different results. Every feature and pipeline stage is strictly deterministic and backed by rigorous tests.

## Core Guarantees

* **Reproducible Builds:** The same inputs and configuration will always produce the exact same output bytes, regardless of worker counts, crash history, or system differences.

* **Integrity:** Every file is hashed and verified. Flipped bytes or truncated files are caught immediately.

* **Crash-Safe:** If a process is killed (e.g., via SIGKILL) mid-build, no partial files are left behind. Rerunning the build will seamlessly pick up where it left off and produce the exact same final bytes.

* **Exact Resume:** During training, if the data loader is interrupted, it can resume from a checkpoint and serve the exact same sequence of batches as an uninterrupted run.

* **Pure Sample Order:** Data shuffling and mixing are pure mathematical functions, completely independent of the number of GPUs or background threads.

## The Pipeline

The tool runs in seven strict, pure-function stages:

1. **download:** Fetches the Wikipedia dump and index safely, with resume support.

2. **extract:** Unzips and parses the XML stream into raw JSON lines.

3. **clean:** Strips wiki markup, filters out unwanted sections, and normalizes text.

4. **tokenize:** Converts text into integer tokens using byte-level BPE (e.g., tiktoken or HuggingFace).

5. **select:** Deduplicates documents and splits them into training and validation sets.

6. **pack:** Packs sequences of tokens into fixed-length rows and chunks them into `.bin` shard files.

7. **verify:** Performs a rigorous integrity check on the final dataset.

## Quickstart (CLI)

`wiki2shards` provides a `typer`-based CLI for running individual stages or the entire build.

```
# Run the entire pipeline based on a config file
wiki2shards build --config configs/simplewiki.yaml

# Verify an existing dataset
wiki2shards verify data/datasets/simplewiki-20260901-gpt2-s2048 --full

# Inspect a specific row in the dataset
wiki2shards inspect data/datasets/simplewiki-20260901-gpt2-s2048 --split train --row 0 --decode

# See how sources are mixed for training
wiki2shards mix-plan --mix configs/mix_example.yaml --steps 100 --global-batch 256

```