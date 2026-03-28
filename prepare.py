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


# ---------------------------------------------------------------------------
# Parsing and caching
# ---------------------------------------------------------------------------

def load_splits() -> dict:
    """Read chain_set_splits.json and return train/validation/test name lists."""
    path = os.path.join(RAW_DIR, "chain_set_splits.json")
    with open(path, "r") as f:
        splits = json.load(f)
    return {
        "train": splits["train"],
        "validation": splits["validation"],
        "test": splits["test"],
    }


def compute_virtual_cb(coords: torch.Tensor) -> torch.Tensor:
    """Compute virtual Cb from N-CA-C geometry.

    Args:
        coords: (L, 4, 3) tensor — N, CA, C, O coordinates.

    Returns:
        (L, 5, 3) tensor — N, CA, C, O, Cb concatenated along atom dim.
    """
    n = coords[:, 0]   # (L, 3)
    ca = coords[:, 1]  # (L, 3)
    c = coords[:, 2]   # (L, 3)
    b = ca - n
    c_vec = c - ca
    a = torch.cross(b, c_vec, dim=-1)
    cb = ca + (-0.58273431 * b + 0.56802827 * c_vec - 0.54067466 * a / (a.norm(dim=-1, keepdim=True) + 1e-7))
    return torch.cat([coords, cb.unsqueeze(1)], dim=1)  # (L, 5, 3)


def build_knn_graph(ca_coords: torch.Tensor, k: int = MAX_NEIGHBORS) -> torch.Tensor:
    """Build k-NN graph on CA coordinates.

    Args:
        ca_coords: (L, 3) tensor of CA positions.
        k: number of nearest neighbors.

    Returns:
        knn_indices: (L, k) long tensor of neighbor indices.
    """
    L = ca_coords.shape[0]
    actual_k = min(k, L - 1)

    dists = torch.cdist(ca_coords.unsqueeze(0), ca_coords.unsqueeze(0)).squeeze(0)  # (L, L)
    # Set self-distance to inf so self is excluded
    dists.fill_diagonal_(float('inf'))

    # Get top-k nearest neighbors
    _, indices = dists.topk(actual_k, dim=-1, largest=False)  # (L, actual_k)

    if actual_k < k:
        # Pad with zeros
        pad = torch.zeros(L, k - actual_k, dtype=torch.long)
        indices = torch.cat([indices, pad], dim=-1)

    return indices.long()


def cache_dataset() -> None:
    """Parse chain_set.jsonl and cache each protein as a .pt file."""
    # Ensure data is downloaded
    download_data()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    jsonl_path = os.path.join(RAW_DIR, "chain_set.jsonl")
    count = 0
    skipped = 0

    with open(jsonl_path, "r") as f:
        for line_idx, line in enumerate(f):
            entry = json.loads(line)
            name = entry["name"]

            # Skip if already cached
            out_path = os.path.join(PROCESSED_DIR, f"{name}.pt")
            if os.path.exists(out_path):
                count += 1
                if count % 1000 == 0:
                    print(f"[cache] Progress: {count} proteins processed (line {line_idx + 1})")
                continue

            seq_str = entry["seq"]
            coord_data = entry["coords"]

            L = len(seq_str)

            # Skip proteins outside length bounds
            if L > MAX_SEQ_LEN or L < 10:
                skipped += 1
                continue

            # Convert sequence to indices, check for unknown AAs
            seq_indices = []
            has_unknown = False
            for aa in seq_str:
                if aa not in AA_TO_IDX:
                    has_unknown = True
                    break
                seq_indices.append(AA_TO_IDX[aa])
            if has_unknown:
                skipped += 1
                continue

            # Parse coordinates: N, CA, C, O — each is list of [x, y, z]
            raw_n = coord_data["N"]
            raw_ca = coord_data["CA"]
            raw_c = coord_data["C"]
            raw_o = coord_data["O"]

            # Build (L, 4, 3) tensor, replacing None/null with NaN
            coords_np = np.full((L, 4, 3), float('nan'), dtype=np.float32)
            for i in range(L):
                for atom_idx, atom_list in enumerate([raw_n, raw_ca, raw_c, raw_o]):
                    xyz = atom_list[i]
                    if xyz is not None and all(v is not None for v in xyz):
                        coords_np[i, atom_idx] = xyz

            coords = torch.from_numpy(coords_np)  # (L, 4, 3)

            # Mask: True where all 4 atoms are valid (no NaN)
            mask = ~torch.isnan(coords).any(dim=-1).any(dim=-1)  # (L,)

            # For virtual Cb and kNN, replace NaN with 0 to avoid propagation
            coords_clean = coords.clone()
            coords_clean[torch.isnan(coords_clean)] = 0.0

            # Compute virtual Cb
            coords_5 = compute_virtual_cb(coords_clean)  # (L, 5, 3)

            # Build kNN graph on CA coordinates
            ca_coords = coords_clean[:, 1]  # (L, 3)
            knn_indices = build_knn_graph(ca_coords)  # (L, k)

            # Save
            torch.save({
                'coords': coords_5,
                'seq': torch.tensor(seq_indices, dtype=torch.long),
                'mask': mask,
                'knn_indices': knn_indices,
                'length': L,
            }, out_path)

            count += 1
            if count % 1000 == 0:
                print(f"[cache] Progress: {count} proteins processed (line {line_idx + 1})")

    print(f"[cache] Done. Total cached: {count}, skipped: {skipped}")
