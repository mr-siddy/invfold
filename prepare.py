"""Protein inverse folding data preparation.

This module provides constants, data download, parsing, featurization,
and data loading utilities for training and evaluating a protein inverse
folding model on the CATH dataset.
"""

import os
import sys
import time
import math
import json
import requests
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 500
TIME_BUDGET = 300
SEED = 42
NUM_AMINO_ACIDS = 20
MAX_NEIGHBORS = 48
NUM_RBF = 16
RBF_D_MIN = 0.0
RBF_D_MAX = 20.0
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
CACHE_DIR = os.path.expanduser("~/.cache/auto-bio/invfold")
RAW_DIR = os.path.join(CACHE_DIR, "raw")
PROCESSED_DIR = os.path.join(CACHE_DIR, "processed")
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_VOCAB)}

CATH_BASE_URL = "https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath"
CATH_FILES = ["chain_set.jsonl", "chain_set_splits.json"]

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(max_retries: int = 3) -> None:
    """Download CATH dataset files to RAW_DIR.

    Downloads ``chain_set.jsonl`` and ``chain_set_splits.json`` from the MIT
    hosted URL.  Skips files that already exist locally.  On failure, retries
    up to *max_retries* times with exponential back-off.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    for filename in CATH_FILES:
        dest = os.path.join(RAW_DIR, filename)
        if os.path.exists(dest):
            print(f"[download] {filename} already exists, skipping.")
            continue

        url = f"{CATH_BASE_URL}/{filename}"
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[download] Downloading {filename} (attempt {attempt}/{max_retries})...")
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()

                # Stream to a temporary file, then rename for atomicity.
                tmp_dest = dest + ".tmp"
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(tmp_dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(f"\r[download] {filename}: {pct}%", end="", flush=True)
                if total:
                    print()  # newline after progress

                os.rename(tmp_dest, dest)
                print(f"[download] {filename} saved ({downloaded:,} bytes).")
                break  # success
            except Exception as exc:
                print(f"\n[download] Attempt {attempt} failed: {exc}")
                # Clean up partial temp file
                tmp_dest = dest + ".tmp"
                if os.path.exists(tmp_dest):
                    os.remove(tmp_dest)
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"[download] Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[download] Failed to download {filename} after {max_retries} attempts.")
                    raise
