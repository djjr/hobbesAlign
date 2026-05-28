"""
Welfare monitor server for The Unknown Room classroom game.

Each player's phone app POSTs {name, round, morsels} after committing a round.
This server tracks current state and serves a facilitator dashboard at GET /.

Run locally:
  uvicorn server:app --reload --host 0.0.0.0 --port 8000

Deploy to Railway:
  railway init && railway up
  (set start command: uvicorn server:app --host 0.0.0.0 --port $PORT)

Deploy to Render:
  New Web Service → connect repo → root dir: welfare_server
  Start command: uvicorn server:app --host 0.0.0.0 --port $PORT

Once deployed, share this URL with students:
  https://your-webapp-url/?server=https://your-server-url
"""
from __future__ import annotations

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

N_PLAYERS = 30

players: dict[str, dict] = {}       # name -> {morsels, round, ts}
round_welfare: dict[int, float] = {} # round -> welfare at time of last report for that round


class Report(BaseModel):
    name: str
    round: int
    morsels: int


@app.post("/report")
def report(r: Report):
    players[r.name] = {"morsels": r.morsels, "round": r.round, "ts": time.time()}
    total = sum(p["morsels"] for p in players.values())
    round_welfare[r.round] = round(total / N_PLAYERS, 3)
    return {"ok": True, "welfare": round_welfare[r.round]}


@app.get("/state")
def get_state():
    total = sum(p["morsels"] for p in players.values())
    welfare = round(total / N_PLAYERS, 3)
    current_round = max((p["round"] for p in players.values()), default=1) if players else 1
    reporters_this_round = sum(1 for p in players.values() if p["round"] == current_round)
    return {
        "welfare": welfare,
        "total_morsels": total,
        "reporters": len(players),
        "reporters_this_round": reporters_this_round,
        "current_round": current_round,
        "players": {name: {"morsels": p["morsels"], "round": p["round"]} for name, p in players.items()},
        "history": round_welfare,
    }


@app.post("/reset")
def reset():
    players.clear()
    round_welfare.clear()
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Welfare Monitor — The Unknown Room</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
      --text: #f1f5f9; --dim: #94a3b8;
      --green: #22c55e; --yellow: #f59e0b; --red: #ef4444;
    }
    body {
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      padding: 32px 24px; max-width: 960px; margin: 0 auto;
    }
    h1 { font-size: 0.8rem; color: var(--dim); text-transform: uppercase;
         letter-spacing: 0.1em; margin-bottom: 24px; }
    .welfare-big { font-size: 6rem; font-weight: 900; line-height: 1; }
    .welfare-sub { font-size: 0.8rem; color: var(--dim); margin-top: 6px; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 28px 0; }
    .card { background: var(--surface); border-radius: 12px; padding: 16px 20px; }
    .card-val { font-size: 2.2rem; font-weight: 900; }
    .card-lbl { font-size: 0.7rem; color: var(--dim); text-transform: uppercase;
                letter-spacing: 0.08em; margin-top: 4px; }
    .sec { font-size: 0.7rem; color: var(--dim); text-transform: uppercase;
           letter-spacing: 0.08em; margin: 24px 0 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--surface2); }
    th { color: var(--dim); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; }
    .bar { font-size: 0.65rem; letter-spacing: -1.5px; }
    .stale { opacity: 0.45; }
    .btn-reset {
      margin-top: 32px; background: rgba(239,68,68,0.12);
      border: 1px solid var(--red); color: var(--red);
      padding: 9px 22px; border-radius: 8px; cursor: pointer;
      font-size: 0.85rem; font-weight: 700; letter-spacing: 0.02em;
    }
    .btn-reset:hover { background: rgba(239,68,68,0.22); }
    .pulse { animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }
  </style>
</head>
<body>
  <h1>The Unknown Room &mdash; Welfare Monitor</h1>

  <div class="welfare-big" id="welfare-num">—</div>
  <div class="welfare-sub">collective welfare &nbsp;(total morsels &divide; 30)</div>

  <div class="grid">
    <div class="card">
      <div class="card-val" id="stat-morsels">—</div>
      <div class="card-lbl">Total morsels</div>
    </div>
    <div class="card">
      <div class="card-val" id="stat-round">—</div>
      <div class="card-lbl">Current round</div>
    </div>
    <div class="card">
      <div class="card-val" id="stat-reporters">—</div>
      <div class="card-lbl">Reported this round</div>
    </div>
  </div>

  <div class="sec">Welfare history</div>
  <table>
    <thead><tr><th>Round</th><th>Welfare</th><th style="width:60%">Bar</th></tr></thead>
    <tbody id="tbody-history"></tbody>
  </table>

  <div class="sec">Player states</div>
  <table>
    <thead><tr><th>Name</th><th>Morsels</th><th>Round</th></tr></thead>
    <tbody id="tbody-players"></tbody>
  </table>

  <button class="btn-reset" onclick="doReset()">Reset Game</button>

  <script>
    function wColor(w) {
      return w > 0.6 ? '#22c55e' : w > 0.3 ? '#f59e0b' : '#ef4444';
    }
    function mColor(m) {
      return m === 0 ? '#ef4444' : m <= 2 ? '#ef4444' : m <= 5 ? '#f59e0b' : '#22c55e';
    }

    async function poll() {
      try {
        const d = await fetch('/state').then(r => r.json());

        const wEl = document.getElementById('welfare-num');
        wEl.textContent = d.welfare.toFixed(3);
        wEl.style.color = wColor(d.welfare);

        document.getElementById('stat-morsels').textContent   = d.total_morsels;
        document.getElementById('stat-round').textContent     = d.current_round;
        document.getElementById('stat-reporters').textContent = d.reporters_this_round + ' / 30';

        // History — newest first
        const rounds = Object.keys(d.history).map(Number).sort((a, b) => b - a);
        document.getElementById('tbody-history').innerHTML = rounds.map(rnd => {
          const w   = d.history[rnd];
          const col = wColor(w);
          const bar = '█'.repeat(Math.round(w * 30));
          return `<tr>
            <td>${rnd}</td>
            <td style="color:${col};font-weight:800">${w.toFixed(3)}</td>
            <td class="bar" style="color:${col}">${bar}</td>
          </tr>`;
        }).join('');

        // Players — sorted by morsels desc
        document.getElementById('tbody-players').innerHTML = Object.entries(d.players)
          .sort((a, b) => b[1].morsels - a[1].morsels)
          .map(([name, p]) => {
            const stale = p.round < d.current_round;
            const note  = stale ? ` <span style="font-size:0.7rem;color:var(--dim)">(R${p.round})</span>` : '';
            return `<tr class="${stale ? 'stale' : ''}">
              <td>${name}${note}</td>
              <td style="color:${mColor(p.morsels)};font-weight:800">${p.morsels}</td>
              <td style="color:var(--dim)">${p.round}</td>
            </tr>`;
          }).join('');
      } catch (_) {}
    }

    async function doReset() {
      if (!confirm('Reset all player state? This cannot be undone.')) return;
      await fetch('/reset', { method: 'POST' });
      poll();
    }

    poll();
    setInterval(poll, 4000);
  </script>
</body>
</html>"""
