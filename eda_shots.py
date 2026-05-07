#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for the cleaned shots dataset.

Reads: data/clean/shots.csv
Writes figures to: img/
Writes report to: doc/eda_shots.md

Usage:
  python3 eda_shots.py

Notes:
- Assumes StatsBomb pitch coordinates (x: 0→120, y: 0→80). Attacking right with goal center at (120, 40).
- Computes shot distance and shot angle (approximate) using goal width scaled to the StatsBomb y-axis.
"""
from __future__ import annotations

import os
import math
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, 'data', 'clean', 'shots.csv')
IMG_DIR = os.path.join(BASE_DIR, 'img')
DOC_DIR = os.path.join(BASE_DIR, 'doc')
REPORT_MD = os.path.join(DOC_DIR, 'eda_shots.md')

PITCH_LENGTH = 120.0  # StatsBomb units
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
# Scale 7.32m goal width to StatsBomb y-axis (assuming 68m pitch width -> SB width 80)
GOAL_WIDTH_SB = 7.32 / 68.0 * PITCH_WIDTH  # ≈ 8.61
HALF_GOAL_SB = GOAL_WIDTH_SB / 2.0
POST_HIGH_Y = GOAL_Y + HALF_GOAL_SB
POST_LOW_Y = GOAL_Y - HALF_GOAL_SB


def ensure_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(DOC_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    # If the input file is missing, return an empty frame with expected columns
    if not os.path.exists(path):
        cols = ['match_id','event_id','period','timestamp','minute','second','team','player','location_x','location_y','outcome','xg']
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    # Standardize column names to lowercase just in case
    df.columns = [c.strip().lower() for c in df.columns]
    # Basic type conversions
    for col in ['location_x', 'location_y', 'xg']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop rows with missing essential fields
    df = df.dropna(subset=['location_x', 'location_y', 'xg'])
    # Normalize outcome labels (trim, title-case certain variants)
    if 'outcome' in df.columns:
        df['outcome'] = (
            df['outcome'].astype(str)
            .str.strip()
            .replace({
                'Off T': 'Off Target',
                'Saved to Post': 'Saved',
                'Wayward': 'Off Target',
                'Saved Off Target': 'Saved',
            })
        )
    return df


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    # Distance to center of goal
    dx = GOAL_X - df['location_x']
    dy = df['location_y'] - GOAL_Y
    df['shot_distance'] = np.hypot(dx, dy)

    # Shot angle using posts at (GOAL_X, POST_LOW_Y) and (GOAL_X, POST_HIGH_Y)
    # Formula: angle between vectors from shot location to each post
    x0 = df['location_x']
    y0 = df['location_y']

    # Vectors to posts
    v1x = GOAL_X - x0
    v1y = POST_LOW_Y - y0
    v2x = GOAL_X - x0
    v2y = POST_HIGH_Y - y0

    # Compute angle via vector formula: theta = arccos( (v1·v2) / (|v1||v2|) )
    dot = v1x * v2x + v1y * v2y
    n1 = np.hypot(v1x, v1y)
    n2 = np.hypot(v2x, v2y)
    # Prevent division by zero
    cos_theta = np.clip(dot / (n1 * n2 + 1e-9), -1.0, 1.0)
    df['shot_angle'] = np.degrees(np.arccos(cos_theta))

    # Bound and clean
    df.loc[~np.isfinite(df['shot_distance']), 'shot_distance'] = np.nan
    df.loc[~np.isfinite(df['shot_angle']), 'shot_angle'] = np.nan
    return df


def summary_stats(df: pd.DataFrame) -> str:
    total = len(df)
    goals = (df['outcome'] == 'Goal').sum() if 'outcome' in df.columns else np.nan
    goal_rate = goals / total if total > 0 else float('nan')
    avg_xg = df['xg'].mean()
    median_xg = df['xg'].median()
    avg_dist = df['shot_distance'].mean()
    med_dist = df['shot_distance'].median()
    out = [
        f"Total shots: {total:,}",
        f"Goals: {goals:,} ({goal_rate:.1%})",
        f"Avg xG: {avg_xg:.3f} | Median xG: {median_xg:.3f}",
        f"Avg distance: {avg_dist:.2f} | Median distance: {med_dist:.2f} (SB units)",
    ]
    return "\n".join(out)


def plot_outcome_distribution(df: pd.DataFrame, outfile: str):
    plt.figure(figsize=(8, 5))
    order = df['outcome'].value_counts().index
    sns.countplot(data=df, x='outcome', order=order, palette='Set2')
    plt.title('Shot outcomes distribution')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_xg_hist(df: pd.DataFrame, outfile: str):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['xg'], bins=50, kde=True, color='#1f77b4')
    plt.title('xG distribution')
    plt.xlabel('xG')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def draw_pitch(ax, color='#222', lw=0.5):
    # Basic StatsBomb pitch outline
    ax.set_facecolor('#f8f9fb')
    # Outer boundaries
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0], [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], color=color, lw=lw)
    # Penalty box (right side)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH-18, PITCH_LENGTH-18, PITCH_LENGTH], [18, 18, PITCH_WIDTH-18, PITCH_WIDTH-18,], color=color, lw=lw)
    # Six-yard box
    ax.plot([PITCH_LENGTH, PITCH_LENGTH-6, PITCH_LENGTH-6, PITCH_LENGTH], [30, 30, PITCH_WIDTH-30, PITCH_WIDTH-30], color=color, lw=lw)
    # Goal line
    ax.plot([PITCH_LENGTH, PITCH_LENGTH+1], [GOAL_Y-HALF_GOAL_SB, GOAL_Y-HALF_GOAL_SB], color=color, lw=lw)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH+1], [GOAL_Y+HALF_GOAL_SB, GOAL_Y+HALF_GOAL_SB], color=color, lw=lw)
    ax.set_xlim(60, 122)  # focus on attacking half
    ax.set_ylim(-2, 82)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def plot_shot_hexbin(df: pd.DataFrame, outfile: str, title: str = 'Shot location density (hexbin)'):
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_pitch(ax)
    hb = ax.hexbin(df['location_x'], df['location_y'], gridsize=30, cmap='mako', mincnt=1, extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH])
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Shots')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_shot_scatter_xg(df: pd.DataFrame, outfile: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_pitch(ax)
    sc = ax.scatter(df['location_x'], df['location_y'], c=df['xg'], s=6, cmap='viridis', alpha=0.6)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('xG')
    ax.set_title('Shot locations colored by xG')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_xg_vs_distance(df: pd.DataFrame, outfile: str):
    # Bin distances and compute mean xG per bin
    bins = np.linspace(0, 40, 41)  # 1-unit bins up to 40 SB units (~penalty area focus)
    labels = (bins[:-1] + bins[1:]) / 2
    s = pd.cut(df['shot_distance'], bins=bins, include_lowest=True)
    mean_xg = df.groupby(s)['xg'].mean().reindex(s.cat.categories)

    plt.figure(figsize=(8, 5))
    plt.plot(labels, mean_xg.values, marker='o', lw=1.5)
    plt.title('Average xG vs shot distance (binned)')
    plt.xlabel('Distance to goal center (SB units)')
    plt.ylabel('Average xG')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_calibration(df: pd.DataFrame, outfile: str):
    # Create binary goal flag and calibrate xG
    y_true = (df['outcome'] == 'Goal').astype(int)
    xg = df['xg']

    # Bin by predicted xG
    bins = np.linspace(0, 1, 21)
    cats = pd.cut(xg, bins=bins, include_lowest=True)
    pred = xg.groupby(cats).mean().reindex(cats.cat.categories)
    obs = y_true.groupby(cats).mean().reindex(cats.cat.categories)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    plt.plot(pred.values, obs.values, marker='o', lw=1.5, label='Empirical')
    plt.title('xG calibration (binned)')
    plt.xlabel('Predicted xG (bin mean)')
    plt.ylabel('Observed goal rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def generate_report(df: pd.DataFrame, images: dict[str, str], out_md: str):
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    stats_block = summary_stats(df)
    md = f"""
    # Shots EDA

    Generated: {now}

    This report summarizes the exploratory analysis of the cleaned shots dataset (StatsBomb events filtered to shots).

    ## Key summary

    {stats_block}

    ## Visualizations

    ![Outcomes]({images['outcomes']})

    ![xG histogram]({images['xg_hist']})

    ![Shot hexbin]({images['hexbin_all']})

    ![Goals hexbin]({images['hexbin_goals']})

    ![Shot scatter xG]({images['scatter_xg']})

    ![xG vs distance]({images['xg_vs_distance']})

    ![xG calibration]({images['calibration']})

    ### Notes
    - Pitch uses StatsBomb dimensions (120 x 80). Right-side goal at x=120.
    - Distance and angle are approximate but consistent across all shots.
    - Calibration is computed by binning predicted xG and comparing to observed goal rate in each bin.

    ### Reproduce
    From the project root:
    
    ```bash
    python3 eda_shots.py
    ```
    Figures will be saved to `img/` and this report to `doc/eda_shots.md`.
    """
    # Dedent and write
    md_text = textwrap.dedent(md).strip() + "\n"
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md_text)


def main():
    ensure_dirs()

    # A consistent style
    sns.set_context('talk')
    sns.set_style('whitegrid')

    df = load_data(DATA_CSV)
    df = add_geometry(df)

    # Produce plots
    images = {}
    images['outcomes'] = os.path.join(IMG_DIR, 'shots_outcomes.png')
    images['xg_hist'] = os.path.join(IMG_DIR, 'shots_xg_hist.png')
    images['hexbin_all'] = os.path.join(IMG_DIR, 'shots_hexbin_all.png')
    images['hexbin_goals'] = os.path.join(IMG_DIR, 'shots_hexbin_goals.png')
    images['scatter_xg'] = os.path.join(IMG_DIR, 'shots_scatter_xg.png')
    images['xg_vs_distance'] = os.path.join(IMG_DIR, 'shots_xg_vs_distance.png')
    images['calibration'] = os.path.join(IMG_DIR, 'shots_calibration.png')

    if 'outcome' in df.columns:
        plot_outcome_distribution(df, images['outcomes'])
    else:
        # Create a placeholder empty figure
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'Outcome column missing', ha='center', va='center')
        plt.axis('off')
        plt.savefig(images['outcomes'], dpi=150)
        plt.close()

    plot_xg_hist(df, images['xg_hist'])
    plot_shot_hexbin(df, images['hexbin_all'], 'Shot location density (all shots)')
    if 'outcome' in df.columns:
        goals_df = df[df['outcome'] == 'Goal']
        if not goals_df.empty:
            plot_shot_hexbin(goals_df, images['hexbin_goals'], 'Shot location density (goals only)')
        else:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'No goals available', ha='center', va='center')
            plt.axis('off')
            plt.savefig(images['hexbin_goals'], dpi=150)
            plt.close()
    else:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'Outcome column missing', ha='center', va='center')
        plt.axis('off')
        plt.savefig(images['hexbin_goals'], dpi=150)
        plt.close()

    plot_shot_scatter_xg(df, images['scatter_xg'])
    plot_xg_vs_distance(df, images['xg_vs_distance'])
    if 'outcome' in df.columns:
        plot_calibration(df, images['calibration'])
    else:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'Outcome column missing', ha='center', va='center')
        plt.axis('off')
        plt.savefig(images['calibration'], dpi=150)
        plt.close()

    # Write report
    generate_report(df, images, REPORT_MD)
    print('EDA complete.')
    print(f"Report: {REPORT_MD}")
    print('Figures:')
    for k, v in images.items():
        print(f"  - {k}: {v}")


if __name__ == '__main__':
    main()
