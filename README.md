# Guardian Angel

A local screen monitoring tool that uses AI to detect explicit content and automatically closes browsers when it is found. For explicitly nude content, it also logs out the current user. Runs continuously in the background, taking periodic screenshots and analyzing them with three stacked ML models.

## How it works

Every 12 seconds, Guardian Angel:

1. Takes a screenshot of the primary monitor
2. Runs it through three detection models in sequence:
   - **Falconsai** — general NSFW image classifier
   - **WD14 ONNX tagger** — detects explicit/questionable anime and illustration content
   - **NudeNet** — fallback detector for exposed and covered body parts (only runs if the first two models don't flag anything)
3. If flagged, all browsers are force-killed (`taskkill`)
4. If the detection is due to **explicit nudity** (WD14 explicit score or NudeNet exposed class), the user is also logged out (`shutdown /l /f`)
5. Detection events are logged to `logs/detections.log` (last 10 entries kept)

Screenshots are never saved to disk. If screen capture fails (e.g. locked screen, UAC prompt), that frame is skipped and monitoring continues.

## Detection thresholds

| Model | Label | Threshold |
|-------|-------|-----------|
| Falconsai | NSFW | 0.40 |
| WD14 | Explicit | 0.10 |
| WD14 | Questionable | 0.25 |
| NudeNet | Exposed classes | 0.45 |
| NudeNet | Covered classes | 0.70 |

## Requirements

- Python 3.10+
- Windows (uses `taskkill`, `shutdown`, and `mss` screen capture)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install mss nudenet pillow transformers huggingface_hub numpy onnxruntime
```

Models are downloaded automatically from Hugging Face on first run.

## Usage

```bash
python main.py
```

Press `Ctrl+C` to stop. Logs print to stdout with timestamps.

## Configuration

All constants are at the top of [main.py](main.py):

| Variable | Default | Description |
|----------|---------|-------------|
| `SCREENSHOT_INTERVAL_SECONDS` | `12` | How often to capture a screenshot |
| `FALCONSAI_THRESHOLD` | `0.40` | Falconsai NSFW score cutoff |
| `ANIME_EXPLICIT_THRESHOLD` | `0.10` | WD14 explicit score cutoff |
| `ANIME_QUESTIONABLE_THRESHOLD` | `0.25` | WD14 questionable score cutoff |
| `NUDENET_EXPOSED_THRESHOLD` | `0.45` | NudeNet exposed class cutoff |
| `NUDENET_COVERED_THRESHOLD` | `0.70` | NudeNet covered class cutoff |
