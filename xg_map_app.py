#!/usr/bin/env python3
"""
Interactive xG map to compare dataset (non-parametric) xG vs the trained MLP model xG.

- Serves a small web app at http://127.0.0.1:5000 (by default)
- Click anywhere on the pitch to see:
  - xG_model: predicted by the trained NN model saved via train_xg_nn.py
  - xG_data: locally weighted average xG from the dataset near the point
- Also shows a background heatmap of the model's xG decision surface.

Usage:
  python3 xg_map_app.py --model-path models/xg_mlp.joblib --host 127.0.0.1 --port 5000

Requirements:
  - Flask (pip install flask)
  - joblib (to load the model)
  - The dataset at data/clean/shots.csv for data-based xG

Notes:
  - If the model file or joblib is missing, the app still runs and shows data xG, but model xG and the heatmap will be unavailable until a model is trained and loadable.
  - Coordinates follow StatsBomb units: x in [0, 120], y in [0, 80], attacking left→right.
"""
from __future__ import annotations

import json
import os
import sys
import argparse
from typing import Optional, Dict, Any

import numpy as np

# Third-party, optional at runtime
try:
    from flask import Flask, request, jsonify, Response
except Exception as e:  # pragma: no cover
    raise RuntimeError("Flask is required to run the interactive app. Install with: pip install flask") from e

try:  # joblib is optional; without it we just disable model predictions
    import joblib  # type: ignore
except Exception as e:  # pragma: no cover
    joblib = None
    _MISSING_JOBLIB_ERR = e

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, 'data', 'clean', 'shots.csv')
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xg_mlp.joblib')

# StatsBomb pitch constants
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Add this folder to sys.path so sibling imports work even when run from elsewhere
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Local imports
from predict_shot import load_shots, predict_shot_metrics  # noqa: E402


def make_app(model_path: str, host: str = '127.0.0.1', port: int = 5000):
    app = Flask(__name__)

    # Load dataset once for fast queries
    shots_df = load_shots(DATA_CSV)

    # Try to load model if available
    model = None
    model_err: Optional[str] = None
    if os.path.exists(model_path) and joblib is not None:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            model_err = f"Failed to load model from {model_path}: {e}"
    else:
        if not os.path.exists(model_path):
            model_err = f"Model not found at {model_path}. Train with: python3 train_xg_nn.py --train"
        elif joblib is None:
            model_err = f"joblib not available to load model: {_MISSING_JOBLIB_ERR}"

    def model_predict_xy(xx: np.ndarray, yy: np.ndarray) -> Optional[np.ndarray]:
        if model is None:
            return None
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        try:
            preds = model.predict(grid)
        except Exception:
            return None
        preds = np.clip(preds, 0.0, 1.0)
        return preds.reshape(xx.shape)

    @app.route('/')
    def index() -> Response:
        # Inline, dependency-free HTML + JS. Uses Canvas to draw pitch and heatmap.
        model_rel = os.path.relpath(model_path, BASE_DIR)
        model_warn_html = f'<div class="row warn">{model_err}</div>' if model_err else ''
        heat_disabled = 'disabled' if model is None else ''
        html = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>xG Interactive Map</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 0; background: #0b132b; color: #eee; }
  header { padding: 12px 16px; background: #1c2541; border-bottom: 1px solid #3a506b; }
  main { display: flex; flex-direction: row; gap: 16px; padding: 16px; }
  #left { flex: 0 0 auto; }
  #right { flex: 1 1 auto; min-width: 260px; }
  #pitch { border: 1px solid #3a506b; background: #173f5f; cursor: crosshair; }
  .panel { background: #1c2541; padding: 12px; border-radius: 6px; border: 1px solid #3a506b; }
  .row { margin: 6px 0; }
  .small { font-size: 12px; color: #cbd5e1; }
  .warn { color: #ffbf69; }
  button { background: #3a506b; color: #fff; border: none; padding: 8px 10px; border-radius: 4px; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  a, a:visited { color: #7bdff2; }
</style>
</head>
<body>
<header>
  <div><strong>xG interactive map</strong> — click the pitch to compare <em>dataset xG</em> vs <em>model xG</em></div>
</header>
<main>
  <div id="left">
    <canvas id="pitch" width="900" height="600"></canvas>
    <div class="small" style="margin-top:8px;">Pitch uses StatsBomb units: x ∈ [0,120], y ∈ [0,80].</div>
  </div>
  <div id="right">
    <div class="panel">
      <div class="row"><strong>Click anywhere</strong> to query xG.</div>
      <div class="row"><span>Last click at:</span> <span id="pt">(—, —)</span></div>
      <div class="row">xG (model): <strong id="xg_model">—</strong></div>
      <div class="row">xG (dataset): <strong id="xg_data">—</strong> <span class="small" id="neighbors"></span></div>
      <div class="row small">Δ (model − data): <span id="delta">—</span></div>
      <div class="row small" id="status"></div>
      <div class="row small">Model file: __MODEL_PATH__</div>
      __MODEL_WARN__
      <hr />
      <div class="row">
        <button id="toggleHeat" __HEAT_DISABLED__>Toggle model heatmap</button>
        <label class="small" style="margin-left:6px;">Resolution:
          <select id="resSel">
            <option value="4" selected>4</option>
            <option value="3">3</option>
            <option value="2">2</option>
          </select>
        </label>
      </div>
    </div>
  </div>
</main>
<script>
(function(){
  const W = 900, H = 600; // 3:2 ratio to match 120:80
  const SBX = 120.0, SBY = 80.0;
  const canvas = document.getElementById('pitch');
  const ctx = canvas.getContext('2d');

  let showHeat = false;
  let heat = null; // {res, nx, ny, values}
  let lastPt = null; // {x,y}

  function sbToPx(x, y) {
    return [x / SBX * W, y / SBY * H];
  }
  function pxToSb(px, py) {
    return [px / W * SBX, py / H * SBY];
  }

  function drawPitchBase() {
    // background
    ctx.fillStyle = '#173f5f';
    ctx.fillRect(0, 0, W, H);
    // border
    ctx.strokeStyle = '#d1e8ff';
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, W-2, H-2);
    // halfway line
    ctx.beginPath();
    ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H); ctx.stroke();
    // center circle
    ctx.beginPath(); ctx.arc(W/2, H/2, 60*(W/120), 0, Math.PI*2); ctx.stroke();
    // penalty spots (approx)
    const pt1 = sbToPx(12, 40); // own half spot
    const pt2 = sbToPx(108, 40);
    const px1 = pt1[0], py1 = pt1[1];
    const px2 = pt2[0], py2 = pt2[1];
    ctx.fillStyle = '#d1e8ff';
    ctx.beginPath(); ctx.arc(px1, py1, 2, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(px2, py2, 2, 0, Math.PI*2); ctx.fill();
  }

  function colorFor(val) {
    // clamp 0..1, simple blue palette
    const v = Math.min(1, Math.max(0, val));
    const rr = Math.round(255 * (0.1 + 0.9*(v))); // more vivid as xG rises
    const gg = Math.round(255 * (0.95 - 0.9*(v)));
    const bb = Math.round(255 * (1 - 0.8*(v)));
    return 'rgb(' + rr + ',' + gg + ',' + bb + ')';
  }

  function drawHeat() {
    if (!showHeat || !heat) return;
    const stepX = heat.res / SBX * W;
    const stepY = heat.res / SBY * H;
    let i = 0;
    for (let yi=0; yi<heat.ny; yi++) {
      for (let xi=0; xi<heat.nx; xi++) {
        const v = heat.values[i++];
        if (v == null) continue;
        ctx.fillStyle = colorFor(v);
        ctx.fillRect(xi*stepX, yi*stepY, Math.ceil(stepX), Math.ceil(stepY));
      }
    }
  }

  function drawMarker() {
    if (!lastPt) return;
    const pt = sbToPx(lastPt.x, lastPt.y);
    const px = pt[0], py = pt[1];
    ctx.strokeStyle = '#ffd166';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI*2); ctx.stroke();
  }

  function render() {
    drawPitchBase();
    drawHeat();
    drawMarker();
  }

  async function fetchHeat(res) {
    const r = await fetch('/api/grid_model?res=' + res);
    if (!r.ok) return null;
    const data = await r.json();
    return data;
  }

  async function updateHeat() {
    const sel = document.getElementById('resSel');
    const res = parseFloat(sel.value);
    const data = await fetchHeat(res);
    if (data && data.values && data.values.length) {
      heat = data;
    } else {
      heat = null;
    }
    render();
  }

  async function onClick(ev) {
    const rect = canvas.getBoundingClientRect();
    const px = ev.clientX - rect.left;
    const py = ev.clientY - rect.top;
    const pt = pxToSb(px, py);
    const x = pt[0], y = pt[1];
    lastPt = {x: x, y: y};
    document.getElementById('pt').textContent = '(' + x.toFixed(2) + ', ' + y.toFixed(2) + ')';
    document.getElementById('status').textContent = 'Querying…';
    try {
      const r = await fetch('/api/predict?x=' + x + '&y=' + y);
      const data = await r.json();
      const xm = data.xg_model;
      const xd = data.xg_data;
      document.getElementById('xg_model').textContent = (xm==null? '—' : xm.toFixed(3));
      document.getElementById('xg_data').textContent = (xd==null? '—' : xd.toFixed(3));
      const delta = (xm!=null && xd!=null) ? (xm - xd) : null;
      document.getElementById('delta').textContent = (delta==null? '—' : (delta>=0? '+' : '') + delta.toFixed(3));
      const nb = (data.neighbors_used!=null) ? ('neighbors_used=' + data.neighbors_used) : '';
      document.getElementById('neighbors').textContent = nb;
      document.getElementById('status').textContent = data.message || '';
    } catch (e) {
      document.getElementById('status').textContent = 'Error querying API';
    }
    render();
  }

  document.getElementById('toggleHeat').addEventListener('click', async () => {
    showHeat = !showHeat;
    if (showHeat && !heat) { await updateHeat(); } else { render(); }
  });
  document.getElementById('resSel').addEventListener('change', async () => {
    if (showHeat) await updateHeat();
  });
  canvas.addEventListener('click', onClick);

  render();
})();
</script>
</body>
</html>
"""
        html = html.replace("__MODEL_PATH__", model_rel)
        html = html.replace("__MODEL_WARN__", model_warn_html)
        html = html.replace("__HEAT_DISABLED__", heat_disabled)
        return Response(html, mimetype='text/html')

    @app.get('/api/predict')
    def api_predict():
        try:
            x = float(request.args.get('x', 'nan'))
            y = float(request.args.get('y', 'nan'))
        except ValueError:
            return jsonify({'error': 'invalid x or y'}), 400

        resp: Dict[str, Any] = {
            'x': x,
            'y': y,
            'xg_model': None,
            'xg_data': None,
            'neighbors_used': None,
            'message': ''
        }

        # Model prediction
        if model is not None:
            try:
                mp = float(np.clip(model.predict(np.array([[x, y]], dtype=float))[0], 0.0, 1.0))
                resp['xg_model'] = mp
            except Exception as e:
                resp['message'] += f" Model prediction failed: {e}."
        else:
            resp['message'] += ' Model not loaded.'

        # Dataset-based local estimate
        try:
            m = predict_shot_metrics(x, y, data=shots_df)
            resp['xg_data'] = float(m.get('xg_pred')) if m.get('xg_pred') is not None else None
            resp['neighbors_used'] = int(m.get('neighbors_used')) if m.get('neighbors_used') is not None else None
        except Exception as e:
            resp['message'] += f" Data xG failed: {e}."

        return jsonify(resp)

    @app.get('/api/grid_model')
    def api_grid_model():
        if model is None:
            return jsonify({'error': 'model not loaded'}), 400
        try:
            res = float(request.args.get('res', '4'))
            if res <= 0:
                res = 4.0
        except Exception:
            res = 4.0
        xs = np.arange(0.0, PITCH_LENGTH + 1e-9, res, dtype=float)
        ys = np.arange(0.0, PITCH_WIDTH + 1e-9, res, dtype=float)
        xx, yy = np.meshgrid(xs, ys)
        zz = model_predict_xy(xx, yy)
        values = None if zz is None else zz.astype(float).ravel().tolist()
        out = {
            'res': res,
            'nx': int(len(xs)),
            'ny': int(len(ys)),
            'values': values or [],
        }
        return jsonify(out)

    # Attach simple helpers to app for potential reuse/testing
    app.shots_df = shots_df  # type: ignore
    app.model = model  # type: ignore

    return app


def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description='Serve an interactive xG map web app.')
    p.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH,
                   help=f'Path to trained model joblib (default: {DEFAULT_MODEL_PATH})')
    p.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind (default: 127.0.0.1)')
    p.add_argument('--port', type=int, default=5000, help='Port to bind (default: 5000)')
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    app = make_app(args.model_path, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    print("xG interactive map server starting…")
    print(f"Open {url} in your browser.")
    if not os.path.exists(args.model_path):
        print(f"Note: model not found at {args.model_path}. Train one with:")
        print("  python3 train_xg_nn.py --train")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
