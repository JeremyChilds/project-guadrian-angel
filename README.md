# Guardian Angel

A local screen monitoring tool that uses AI to detect explicit content and automatically closes browsers when it is found. Runs continuously in the background, taking periodic screenshots and analyzing them with three stacked ML models.

## How it works

Every 8 seconds, Guardian Angel:

1. Takes a screenshot of the primary monitor
2. Runs it through three detection models in sequence:
   - **Falconsai** — general NSFW image classifier (HuggingFace)
   - **WD14 ONNX tagger** — detects explicit/questionable anime/illustration content
   - **NudeNet** — detects exposed and covered body parts
3. If any model flags the image, all browsers are force-killed (`taskkill`)
4. Clean screenshots are deleted immediately; flagged screenshots are kept in `logs/` for up to 7 days

## Detection thresholds

| Model | Label | Threshold |
|-------|-------|-----------|
| Falconsai | NSFW | 0.40 |
| WD14 | Explicit | 0.10 |
| WD14 | Questionable | 0.15 |
| NudeNet | Exposed classes | 0.45 |
| NudeNet | Covered classes | 0.70 |

## Requirements

- Python 3.10+
- Windows (uses `taskkill` and `mss` screen capture)

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
| `SCREENSHOT_INTERVAL_SECONDS` | `8` | How often to capture a screenshot |
| `RETENTION_DAYS` | `7` | How long flagged screenshots are kept |
| `LOGS_DIR` | `logs/` | Where flagged screenshots are saved |
| `FALCONSAI_THRESHOLD` | `0.40` | Falconsai NSFW score cutoff |
| `ANIME_EXPLICIT_THRESHOLD` | `0.10` | WD14 explicit score cutoff |
| `ANIME_QUESTIONABLE_THRESHOLD` | `0.15` | WD14 questionable score cutoff |
| `NUDENET_EXPOSED_THRESHOLD` | `0.45` | NudeNet exposed class cutoff |
| `NUDENET_COVERED_THRESHOLD` | `0.70` | NudeNet covered class cutoff |
