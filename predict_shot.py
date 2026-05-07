#!/usr/bin/env python3
"""
Location-based shot outcome metrics and xG estimator.

Given a StatsBomb-style shot location (x, y) in SB units (x: 0→120, y: 0→80),
this module estimates the following metrics non-parametrically from the
historical shots dataset in data/clean/shots.csv:

- distance_to_goal: Euclidean distance to the goal center at (120, 40)
- xg_pred: locally-weighted average of StatsBomb xG of nearby shots
- outcome_probs: likelihoods for outcomes in {Off Target, Blocked, Saved, Goal, Post}
- shot_density_per_sbunit2: local shot density (count / area) near the query point
- goal_density_per_sbunit2: local goal density (count / area) near the query point

Usage (CLI):
  python3 predict_shot.py --x 102 --y 40 --radius 6 --min-neighbors 200

Programmatic:
  from predict_shot import predict_shot_metrics
  m = predict_shot_metrics(102, 40)
  print(m["xg_pred"], m["outcome_probs"]) 

Notes:
- Uses a Gaussian kernel (bandwidth = radius / 2 by default) for weighted
  averages. Falls back to k-nearest neighbors if the radius window is too sparse.
- Paths are anchored to this file location, so it works regardless of CWD.
- If the cleaned CSV is missing or empty, the function returns NaNs/zeros with
  neighbors_used = 0 and an informative message printed to stderr.
"""
from __future__ import annotations

import json
import math
import os
import sys
import argparse
from typing import Dict, Optional

import numpy as np
import pandas as pd

# --- Constants aligned with eda_shots.py ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, 'data', 'clean', 'shots.csv')

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
# Goal width projected to SB y-axis (not essential here, but kept for parity)
GOAL_WIDTH_SB = 7.32 / 68.0 * PITCH_WIDTH
HALF_GOAL_SB = GOAL_WIDTH_SB / 2.0

# Outcome canonical set we will report probabilities for
CANON_OUTCOMES = [
    'Off Target',
    'Blocked',
    'Saved',
    'Goal',
    'Post',
]

# Outcome normalization mapping to canonical set
_OUTCOME_MAP = {
    'off t': 'Off Target',
    'off target': 'Off Target',
    'wayward': 'Off Target',
    'blocked': 'Blocked',
    'saved': 'Saved',
    'saved off target': 'Saved',
    'saved to post': 'Saved',
    'goal': 'Goal',
    'post': 'Post',
}


def _normalize_outcome(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    key = str(x).strip().lower()
    return _OUTCOME_MAP.get(key, x if x in CANON_OUTCOMES else x)


def load_shots(path: str = DATA_CSV) -> pd.DataFrame:
    """Load cleaned shots CSV with light normalization.

    Returns a DataFrame with numeric location_x, location_y, xg and a normalized
    'outcome' column mapped to the canonical outcomes when possible. Invalid
    rows (missing locations) are dropped.
    """
    if not os.path.exists(path):
        # Return empty with expected columns
        cols = ['match_id','event_id','period','timestamp','minute','second',
                'team','player','location_x','location_y','outcome','xg']
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Coerce numeric columns
    for col in ['location_x', 'location_y', 'xg']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows without locations; keep rows even if xg is NaN
    df = df.dropna(subset=['location_x', 'location_y'])

    # Normalize outcomes
    if 'outcome' in df.columns:
        df['outcome'] = df['outcome'].map(_normalize_outcome)

    return df


def distance_to_goal(x: float, y: float) -> float:
    return float(math.hypot(GOAL_X - x, y - GOAL_Y))


def _gaussian_weights(d: np.ndarray, bandwidth: float) -> np.ndarray:
    bw = max(float(bandwidth), 1e-6)
    w = np.exp(-(d ** 2) / (2.0 * bw ** 2))
    return w


def _effective_window(distances: np.ndarray, radius: float, min_neighbors: int) -> tuple[np.ndarray, float]:
    """Return indices of neighbors within radius (or kNN fallback) and the effective radius used.
    """
    within = np.where(distances <= radius)[0]
    if within.size >= min_neighbors:
        return within, float(radius)
    # Fallback to kNN with expanded radius to k-th neighbor
    if distances.size == 0:
        return within, float(radius)
    k = min(min_neighbors, distances.size)
    # Argpartition to find k-th smallest
    kth = np.partition(distances, k - 1)[k - 1]
    eff_r = max(float(radius), float(kth))
    knn_idx = np.argsort(distances)[:k]
    return knn_idx, eff_r


def predict_shot_metrics(
    x: float,
    y: float,
    radius: float = 6.0,
    min_neighbors: int = 200,
    bandwidth: Optional[float] = None,
    data: Optional[pd.DataFrame] = None,
) -> Dict:
    """Estimate local shot metrics at (x, y) using historical shots.

    Parameters:
    - x, y: Shot location in StatsBomb units.
    - radius: Window radius (SB units) for local neighborhood.
    - min_neighbors: Minimum neighbors; if not met within radius, fall back to
      k-nearest neighbors with effective radius expanded to include the k-th.
    - bandwidth: Gaussian kernel bandwidth; default is radius/2.
    - data: Optional pre-loaded DataFrame (from load_shots). If None, it will
      be loaded from disk.

    Returns:
      Dict with keys: x, y, distance_to_goal, xg_pred, outcome_probs (dict),
      shot_density_per_sbunit2, goal_density_per_sbunit2, neighbors_used,
      effective_radius.
    """
    df = data if data is not None else load_shots()
    if df is None or df.empty:
        # Graceful empty result
        return {
            'x': float(x),
            'y': float(y),
            'distance_to_goal': distance_to_goal(x, y),
            'xg_pred': float('nan'),
            'outcome_probs': {k: 0.0 for k in CANON_OUTCOMES},
            'shot_density_per_sbunit2': 0.0,
            'goal_density_per_sbunit2': 0.0,
            'neighbors_used': 0,
            'effective_radius': float(radius),
            'note': f'No data available at {DATA_CSV}',
        }

    # Extract arrays
    X = df['location_x'].to_numpy(dtype=float, copy=False)
    Y = df['location_y'].to_numpy(dtype=float, copy=False)
    D = np.hypot((GOAL_X - X), (Y - GOAL_Y))  # not used directly; keep if needed

    # Distances to query
    qdist = np.hypot((X - x), (Y - y))

    idx, eff_r = _effective_window(qdist, radius=radius, min_neighbors=min_neighbors)

    if idx.size == 0:
        return {
            'x': float(x),
            'y': float(y),
            'distance_to_goal': distance_to_goal(x, y),
            'xg_pred': float('nan'),
            'outcome_probs': {k: 0.0 for k in CANON_OUTCOMES},
            'shot_density_per_sbunit2': 0.0,
            'goal_density_per_sbunit2': 0.0,
            'neighbors_used': 0,
            'effective_radius': float(eff_r),
            'note': 'No neighbors found',
        }

    # Window data
    dists = qdist[idx]
    sub = df.iloc[idx]

    # Kernel weights
    bw = float(bandwidth) if bandwidth is not None else max(eff_r / 2.0, 1e-6)
    w = _gaussian_weights(dists, bw)
    w_sum = float(w.sum()) if np.isfinite(w).all() else 0.0

    # xG: weighted average over available xg values
    xg_vals = pd.to_numeric(sub.get('xg'), errors='coerce').to_numpy(dtype=float)
    mask_xg = np.isfinite(xg_vals)
    if mask_xg.any() and w_sum > 0:
        xg_pred = float(np.average(xg_vals[mask_xg], weights=w[mask_xg]))
        # Clip to [0,1] just in case
        xg_pred = float(min(max(xg_pred, 0.0), 1.0))
    else:
        xg_pred = float('nan')

    # Outcome probabilities (weighted frequencies with Laplace smoothing)
    outcomes = sub.get('outcome')
    if outcomes is None:
        probs = {k: 0.0 for k in CANON_OUTCOMES}
    else:
        outs = outcomes.fillna('Unknown').astype(str).map(_normalize_outcome)
        # Build weight per class
        probs = {}
        alpha = 1.0  # Laplace smoothing
        denom = w_sum + alpha * len(CANON_OUTCOMES)
        for k in CANON_OUTCOMES:
            cls_mask = (outs == k).to_numpy()
            cls_w = float(w[cls_mask].sum()) if cls_mask.any() else 0.0
            probs[k] = float((cls_w + alpha) / denom) if denom > 0 else 0.0

    # Densities using unweighted counts per area (within effective radius)
    area = math.pi * (eff_r ** 2)
    # Use counts within the disk of eff_r, not just idx if fallback selected
    disk_mask = qdist <= eff_r
    n_in_area = int(disk_mask.sum())
    goals_in_area = 0
    if 'outcome' in df.columns:
        goals_in_area = int((df.loc[disk_mask, 'outcome'].map(_normalize_outcome) == 'Goal').sum())

    shot_density = (n_in_area / area) if area > 0 else 0.0
    goal_density = (goals_in_area / area) if area > 0 else 0.0

    return {
        'x': float(x),
        'y': float(y),
        'distance_to_goal': distance_to_goal(x, y),
        'xg_pred': xg_pred,
        'outcome_probs': probs,
        'shot_density_per_sbunit2': float(shot_density),
        'goal_density_per_sbunit2': float(goal_density),
        'neighbors_used': int(idx.size),
        'effective_radius': float(eff_r),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Predict local shot metrics at (x, y) in StatsBomb units.')
    p.add_argument('--x', type=float, required=True, help='x coordinate (0-120)')
    p.add_argument('--y', type=float, required=True, help='y coordinate (0-80)')
    p.add_argument('--radius', type=float, default=6.0, help='search radius in SB units (default: 6)')
    p.add_argument('--min-neighbors', type=int, default=200, help='minimum neighbors fallback (default: 200)')
    p.add_argument('--bandwidth', type=float, default=None, help='Gaussian kernel bandwidth; default radius/2')
    p.add_argument('--json', action='store_true', help='output raw JSON on a single line')
    args = p.parse_args(argv)

    res = predict_shot_metrics(
        x=args.x,
        y=args.y,
        radius=args.radius,
        min_neighbors=args.min_neighbors,
        bandwidth=args.bandwidth,
    )

    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        # Pretty print
        print(f"Query: x={res['x']:.2f}, y={res['y']:.2f}")
        print(f"Distance to goal: {res['distance_to_goal']:.2f} (SB units)")
        print(f"Predicted xG: {res['xg_pred'] if np.isfinite(res['xg_pred']) else 'NaN'}")
        print("Outcome probabilities:")
        for k in CANON_OUTCOMES:
            v = res['outcome_probs'].get(k, 0.0)
            print(f"  - {k:11s}: {v:.3f}")
        print(f"Shot density (/SB unit^2): {res['shot_density_per_sbunit2']:.4f}")
        print(f"Goal density (/SB unit^2): {res['goal_density_per_sbunit2']:.5f}")
        print(f"Neighbors used: {res['neighbors_used']} | Effective radius: {res['effective_radius']:.2f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
