#!/usr/bin/env python3
"""
Train a basic neural net (MLP) to predict xG from shot location only.

- Input features: [location_x, location_y] in StatsBomb units (x: 0→120, y: 0→80)
- Target: xg column from the cleaned dataset (already provided by StatsBomb pipeline)
- Model: StandardScaler + MLPRegressor (ReLU), with early stopping
- Output: saves a joblib Pipeline to models/xg_mlp.joblib by default

Usage examples:
  # Train with defaults and save model
  python3 train_xg_nn.py --train

  # Train with custom hidden layers and validation split
  python3 train_xg_nn.py --train --hidden 64 32 --val-size 0.2 --seed 7

  # Predict for a single location using a saved model
  python3 train_xg_nn.py --predict --x 102 --y 40

Notes:
- If the cleaned CSV is missing or has no valid rows, the script will exit gracefully.
- Only location_x and location_y are used as inputs per the requirement; no handcrafted features.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# Optional imports guarded to keep helpful error message if missing
try:
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception as e:  # pragma: no cover
    MISSING_SKLEARN_ERR = e
    train_test_split = None  # type: ignore
    MLPRegressor = None  # type: ignore
    Pipeline = None  # type: ignore
    StandardScaler = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    mean_squared_error = None  # type: ignore
    r2_score = None  # type: ignore

try:
    import joblib
except Exception as e:  # pragma: no cover
    joblib = None
    MISSING_JOBLIB_ERR = e

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, 'data', 'clean', 'shots.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'xg_mlp.joblib')

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def _load_data(path: str = DATA_CSV) -> pd.DataFrame:
    """Load the cleaned shots CSV and return rows with valid x, y, and xg."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=['location_x', 'location_y', 'xg'])

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ['location_x', 'location_y', 'xg']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only rows with valid features and target
    df = df.dropna(subset=['location_x', 'location_y', 'xg']).copy()

    # Clip xg to [0, 1] range just in case
    df['xg'] = df['xg'].clip(0.0, 1.0)
    return df


def _build_model(hidden: Tuple[int, ...], lr: float, alpha: float, max_iter: int, seed: int) -> Pipeline:
    if MLPRegressor is None or Pipeline is None or StandardScaler is None:
        raise RuntimeError(
            'scikit-learn is required. Import error: ' + str(MISSING_SKLEARN_ERR)
        )

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        verbose=False,
        tol=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    )

    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('mlp', mlp),
    ])
    return pipe


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if mean_absolute_error is None:
        return {}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Also provide a simple calibration-like bucket check
    clipped = np.clip(y_pred, 0.0, 1.0)
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'PredMean': float(np.mean(clipped)),
        'TrueMean': float(np.mean(y_true)),
    }


def train(hidden: List[int], val_size: float, seed: int, lr: float, alpha: float, max_iter: int,
          model_path: str) -> dict:
    df = _load_data(DATA_CSV)
    if df.empty:
        print('No training data found: ' + DATA_CSV, file=sys.stderr)
        return {'status': 'no_data'}

    X = df[['location_x', 'location_y']].to_numpy(dtype=float)
    y = df['xg'].to_numpy(dtype=float)

    if train_test_split is None:
        raise RuntimeError('scikit-learn is required for training')

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=val_size, random_state=seed, shuffle=True
    )

    model = _build_model(tuple(hidden), lr=lr, alpha=alpha, max_iter=max_iter, seed=seed)
    model.fit(X_tr, y_tr)

    # Evaluate
    yhat_tr = np.clip(model.predict(X_tr), 0.0, 1.0)
    yhat_va = np.clip(model.predict(X_va), 0.0, 1.0)

    metrics_tr = _evaluate(y_tr, yhat_tr)
    metrics_va = _evaluate(y_va, yhat_va)

    # Ensure output directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if joblib is None:
        raise RuntimeError('joblib is required to save the model. Import error: ' + str(MISSING_JOBLIB_ERR))
    joblib.dump(model, model_path)

    result = {
        'status': 'ok',
        'samples_total': int(len(df)),
        'samples_train': int(len(y_tr)),
        'samples_val': int(len(y_va)),
        'model_path': model_path,
        'hidden_layers': hidden,
        'learning_rate_init': lr,
        'alpha': alpha,
        'max_iter': max_iter,
        'metrics_train': metrics_tr,
        'metrics_val': metrics_va,
    }

    # Pretty print summary
    print('Training summary:')
    for k, v in result.items():
        if isinstance(v, dict):
            print(f'  {k}:')
            for kk, vv in v.items():
                print(f'    - {kk}: {vv}')
        else:
            print(f'  {k}: {v}')

    return result


def predict(x: float, y: float, model_path: str = DEFAULT_MODEL_PATH) -> float:
    if joblib is None:
        raise RuntimeError('joblib is required to load the model. Import error: ' + str(MISSING_JOBLIB_ERR))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}. Train one with --train.')
    model = joblib.load(model_path)
    Xq = np.array([[float(x), float(y)]], dtype=float)
    pred = model.predict(Xq)[0]
    return float(np.clip(pred, 0.0, 1.0))


def _forward_mlp_numpy(xy_scaled: np.ndarray, coefs: List[np.ndarray], intercepts: List[np.ndarray], activation: str = 'relu') -> np.ndarray:
    """Run a forward pass for a single-sample 2D input through exported weights.
    xy_scaled: shape (n_features,) or (1, n_features)
    Returns array of shape (n_outputs,)
    """
    z = xy_scaled.reshape(1, -1)
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        z = z @ W + b  # shape (1, units)
        # Apply activation for all but last layer (regression output is identity)
        if i < len(coefs) - 1:
            if activation == 'relu':
                z = np.maximum(z, 0.0)
            else:
                raise ValueError(f'Unsupported activation: {activation}')
    return z.ravel()


def export_web(model_path: str = DEFAULT_MODEL_PATH, out_path: Optional[str] = None, verify_samples: int = 64, seed: int = 0) -> dict:
    """Export the sklearn Pipeline (StandardScaler + MLPRegressor) to a web-friendly JSON.

    The JSON contains:
      - scaler mean and scale
      - MLP layer sizes, weight matrices, and biases
      - metadata and clipping range

    Returns a dict with summary and the written path.
    """
    if joblib is None:
        raise RuntimeError('joblib is required to load the model. Import error: ' + str(MISSING_JOBLIB_ERR))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}. Train one with --train.')

    model = joblib.load(model_path)

    # Try to access pipeline steps
    try:
        scaler = model.named_steps.get('scaler', None)
        mlp = model.named_steps.get('mlp', None)
    except Exception:
        # If not a Pipeline, assume direct estimator with attributes
        scaler = getattr(model, 'scaler', None)
        mlp = getattr(model, 'mlp', model)

    # Validate components
    if scaler is None or not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        raise ValueError('Expected a StandardScaler named "scaler" in the Pipeline with fitted mean_ and scale_.')
    if mlp is None or not hasattr(mlp, 'coefs_') or not hasattr(mlp, 'intercepts_'):
        raise ValueError('Expected an MLPRegressor named "mlp" in the Pipeline with trained coefs_ and intercepts_.')

    # Collect parameters
    mean = np.asarray(scaler.mean_, dtype=float).tolist()
    scale = np.asarray(scaler.scale_, dtype=float).tolist()
    layers = [int(mlp.n_features_in_)] + [int(s) for s in (mlp.hidden_layer_sizes if isinstance(mlp.hidden_layer_sizes, tuple) else tuple(mlp.hidden_layer_sizes))] + [int(mlp.n_outputs_)]
    coefs = [np.asarray(W, dtype=float).tolist() for W in mlp.coefs_]
    intercepts = [np.asarray(b, dtype=float).tolist() for b in mlp.intercepts_]

    export = {
        'version': 1,
        'framework': 'scikit-learn',
        'pipeline': ['StandardScaler', 'MLPRegressor'],
        'feature_names': ['location_x', 'location_y'],
        'scaler': {
            'mean': mean,
            'scale': scale,
        },
        'mlp': {
            'layers': layers,
            'coefs': coefs,
            'intercepts': intercepts,
            'activation': 'relu',
            'out_activation': 'identity',
        },
        'clip': [0.0, 1.0],
    }

    # Default output path
    if out_path is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        out_path = os.path.join(MODELS_DIR, 'xg_mlp_web.json')

    # Write JSON (compact but readable)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, separators=(',', ':'), ensure_ascii=False)

    # Optional verification: compare sklearn vs exported forward
    rng = np.random.default_rng(seed)
    # Build numpy arrays for coefs
    np_coefs = [np.asarray(W, dtype=float) for W in mlp.coefs_]
    np_intercepts = [np.asarray(b, dtype=float) for b in mlp.intercepts_]

    X_samples = np.column_stack([
        rng.uniform(0.0, PITCH_LENGTH, size=verify_samples),
        rng.uniform(0.0, PITCH_WIDTH, size=verify_samples),
    ])
    # Scaler transform
    Xs = (X_samples - np.asarray(mean)) / np.asarray(scale)

    # Forward pass using exported params
    preds_export = []
    for i in range(verify_samples):
        v = _forward_mlp_numpy(Xs[i], np_coefs, np_intercepts, activation='relu')[0]
        preds_export.append(v)
    preds_export = np.asarray(preds_export)

    # Sklearn pipeline predictions
    preds_sklearn = np.asarray(model.predict(X_samples), dtype=float)

    # Clip both
    preds_export_c = np.clip(preds_export, 0.0, 1.0)
    preds_sklearn_c = np.clip(preds_sklearn, 0.0, 1.0)
    max_abs_err = float(np.max(np.abs(preds_export_c - preds_sklearn_c))) if verify_samples > 0 else 0.0

    print(f'Web export written to: {out_path}')
    print(f'Verification max|diff| on {verify_samples} random samples: {max_abs_err:.6g}')

    return {
        'status': 'ok',
        'out_path': out_path,
        'verify_samples': verify_samples,
        'max_abs_err': max_abs_err,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train/Predict xG using a simple MLP on (x, y).')
    sub = p.add_subparsers(dest='cmd')

    p.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH,
                   help=f'Path to save/load the model (default: {DEFAULT_MODEL_PATH})')

    # Flags without subcommands for quick usage
    p.add_argument('--train', action='store_true', help='Train the model')
    p.add_argument('--predict', action='store_true', help='Predict for a single (x, y)')
    p.add_argument('--export-web', action='store_true', help='Export model to a web-friendly JSON for static browser demos')

    # Training options
    p.add_argument('--hidden', type=int, nargs='+', default=[32, 16],
                   help='Hidden layer sizes, e.g., --hidden 64 32 16 (default: 32 16)')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate init for Adam (default: 1e-3)')
    p.add_argument('--alpha', type=float, default=1e-4, help='L2 regularization (alpha) (default: 1e-4)')
    p.add_argument('--max-iter', type=int, default=500, help='Max training iterations for MLP (default: 500)')
    p.add_argument('--val-size', type=float, default=0.15, help='Validation split size (default: 0.15)')
    p.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    # Prediction options
    p.add_argument('--x', type=float, help='Shot location x (StatsBomb units 0..120)')
    p.add_argument('--y', type=float, help='Shot location y (StatsBomb units 0..80)')

    # Export options
    p.add_argument('--web-out', type=str, default=None, help='Output path for exported web JSON (default: models/xg_mlp_web.json)')
    p.add_argument('--verify-samples', type=int, default=64, help='Random samples to verify export parity (default: 64)')
    p.add_argument('--verify-seed', type=int, default=0, help='Seed for verification sampling (default: 0)')

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Determine command
    cmd = args.cmd
    # Allow top-level boolean flags as shortcuts
    if cmd is None:
        if args.train:
            cmd = 'train'
        elif args.predict:
            cmd = 'predict'
        elif args.export_web:
            cmd = 'export-web'

    if cmd == 'train':
        train(
            hidden=list(args.hidden),
            val_size=float(args.val_size),
            seed=int(args.seed),
            lr=float(args.lr),
            alpha=float(args.alpha),
            max_iter=int(args.max_iter),
            model_path=str(args.model_path),
        )
    elif cmd == 'predict':
        if args.x is None or args.y is None:
            parser.error('--predict requires --x and --y')
        pred = predict(float(args.x), float(args.y), model_path=str(args.model_path))
        print(f'Predicted xG at (x={args.x:.2f}, y={args.y:.2f}): {pred:.4f}')
    elif cmd == 'export-web':
        res = export_web(model_path=str(args.model_path), out_path=str(args.web_out) if args.web_out else None,
                         verify_samples=int(args.verify_samples), seed=int(args.verify_seed))
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
