# 🎮 MoveMatch

> **Follow the avatars. Move on the beat. Let AI score your timing.**

MoveMatch is a Just Dance-style game played in front of a webcam. Pick a choreography, and avatars slide across the top of the screen: when one reaches the line, everyone does that move. A camera films all the players at once, and a local pose model then measures **how early or late each player started each move**.

---

## 🕹️ The Game

| Step | What happens |
|------|--------------|
| **1. Pick** | Choose a choreography (a *script*) by its number of moves: Warm-up (3), Groove (5) or Marathon (8). Optionally type the players' names, left → right. |
| **2. Dance** | The camera goes fullscreen (mirrored), counts down 3-2-1, then starts recording. Each move's avatar enters at the right edge 2 s before its beat and slides left; when it reaches the line (10% from the left), do the move. |
| **3. Score** | Recording stops 1 s after the last beat. The clip is analyzed with local pose estimation: every player is tracked left → right and the start of each move is timed against its beat. |
| **4. Rank** | The results show a per-move table (rating + offset per player), totals, the winner, and a boxed snapshot of them. Named players' scores go to the per-script high-score tables. |

### Scoring

For each move with beat time **T**, the app looks at the player's pose between **T − 1 s** and **T + 1 s**:

- The move's **start** is the first moment the pose appears in that window. If the pose disappears for less than 0.3 s and comes back, the app treats it as the same pose still being held.
- A player who is already in the pose when the window opens gets the earliest allowed start (−1.0 s).
- No start in the window = **Miss**.

The offset (start − T; negative = early) gives the rating:

| \|offset\| | Rating | Points |
|---|---|---|
| ≤ 0.2 s | Perfect | 100 |
| ≤ 0.4 s | Great | 70 |
| ≤ 0.7 s | Good | 40 |
| ≤ 1.0 s | OK | 10 |
| otherwise | Miss | 0 |

The highest total wins. A tie goes to the smaller summed |offset|. These values are constants at the top of `app/dance.py`.

### Moves

Scripts can use the gestures the pose geometry recognizes: `arms_crossed`, `clap`, `hands_on_head`, `raise_one_arm`, `crouch`, `touch_knee`, `point_camera`. (`wave` is also detected, but the detector can't tell it apart from `raise_one_arm`, so scripts don't use it.) Each move has an avatar in `app/static/avatars/` (regenerate and self-check with `python app/avatars.py`).

### Scripts

A script is a JSON file in `app/scripts/`:

```json
{
  "id": "warmup",
  "name": "Warm-up",
  "steps": [
    {"move": "clap", "at": 2.0},
    {"move": "raise_one_arm", "at": 4.0},
    {"move": "hands_on_head", "at": 6.0}
  ]
}
```

`at` is the beat time in seconds from the start of the recording, so beats don't have to be evenly spaced (ready for songs later). Scripts are validated at startup: known moves only, increasing times, no move twice in a row. There is one script per move count, and the Play page lists them by length.

---

## 🚀 Running It

MoveMatch is a small Flask app (`app/`) that runs on Linux, macOS and Windows. It needs **Python ≥ 3.11** and the deps in `pyproject.toml`; the pose model (~13 MB) ships in `models/yolov8n-pose.onnx` (re-downloaded if missing) and runs on the CPU. ffmpeg is used to transcode clips to H.264 MP4: a system `ffmpeg` on `PATH` is used if present, otherwise the binary bundled with `imageio-ffmpeg`.

**Linux/macOS** — one command:

```bash
app/run.sh                       # gunicorn on http://127.0.0.1:5000
```

On first run, `run.sh` creates `.venv/` in the repo root and installs the dependencies into it (with [uv](https://docs.astral.sh/uv/) if installed, which also downloads Python 3.12 when needed; otherwise with the first Python ≥ 3.11 on `PATH`). The dependencies are reinstalled whenever `pyproject.toml` changes.

**Windows (or any OS without gunicorn)** — set up the venv once, then start with waitress:

```bash
python -m venv .venv             # Python >= 3.11
.venv\Scripts\activate           # Linux/macOS: source .venv/bin/activate
pip install .                    # installs the dependencies only
python app/app.py                # waitress on http://127.0.0.1:5000
```

Both read a few environment variables (all optional):

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOST` / `PORT` | `127.0.0.1` / `5000` | bind address (use `HOST=0.0.0.0` to expose on the LAN) |
| `SECRET_KEY` | `dev-change-me` | Flask session secret |
| `KEEP_UPLOADS` | `0` | `1` keeps uploaded clips in `app/uploads/` |
| `MOVEMATCH_VENV` | repo `.venv` (auto-created) | existing virtualenv for `run.sh` to use as-is |
| `WEB_WORKERS` | `2` | gunicorn worker processes (`run.sh` only) |

> Camera recording in the browser needs a secure context: open the app via `http://localhost` / `127.0.0.1`, or put it behind HTTPS when accessing it from another device.

Pages:

- `/` — **Play**: pick a script, dance in front of the camera, see the scores.
- `/leaderboard` — best score per player for each script, and a gallery of boxed winner snapshots.
- `/test` — **Script tester**: upload a clip recorded while following a script (the clip's first frame is the script's t = 0) and see the same results. Nothing is saved. Handy for tuning: record with `KEEP_UPLOADS=1`, then replay the kept clip here.

Results are stored locally in `app/movematch.db` (SQLite) and snapshots in `app/static/snapshots/`.

---

## 🧠 How It Works

| File | Role |
|------|------|
| `app/templates/index.html` | Fullscreen stage: mirrored camera, avatar lane, 3-2-1 countdown. The script clock starts on the recorder's `start` event, so script t = 0 is video t = 0. |
| `app/yolo.py` | 10 times per second, YOLOv8n-pose (ONNX, OpenCV on CPU) finds each person and their 17 body points. People are tracked left → right, and simple geometry names each person's gesture in every sampled frame. |
| `app/dance.py` | Loads and validates the scripts, merges each player's detections into pose runs, and turns each run's start into an offset, rating and points per beat. |
| `app/app.py` | Flask routes, ffmpeg transcode, winner snapshot (taken at the winner's most accurate move), SQLite persistence. |

```text
Browser (fullscreen stage, MediaRecorder)
        │  WebM clip + script id + names
        ▼
Flask app (app/app.py)  ──ffmpeg──▶  H.264 MP4
        │
        ▼
yolo.extract_players  ──▶  per player: {gesture: [times seen]}
        │
        ▼
dance.score_players   ──▶  per player, per beat: offset → rating → points
        │
        ▼
winner snapshot + SQLite (games, scores)  ──▶  results + leaderboard
```

---

## 🗺️ Roadmap

**Next up:**

- 🎵 Songs: scripts synced to music
- ⚡ Real-time (per-frame) feedback while dancing
- ✋ Raise-hand player registration
- 👤 Face recognition to identify and name players automatically

**Watch. Move. Hit the beat. Climb the leaderboard. 🏆**
