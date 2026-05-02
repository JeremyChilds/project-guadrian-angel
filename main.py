import csv
import logging
import subprocess
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import mss
import mss.exception
from nudenet import NudeDetector
from PIL import Image
from transformers import pipeline
from huggingface_hub import hf_hub_download
import numpy as np
from onnxruntime import InferenceSession

SCREENSHOT_INTERVAL_SECONDS = 30
LOGS_DIR = Path(__file__).parent / "logs"
DETECTION_LOG = LOGS_DIR / "detections.log"
CRASH_LOG = LOGS_DIR / "crash.log"

FALCONSAI_THRESHOLD = 0.90
ANIME_EXPLICIT_THRESHOLD = 0.9
ANIME_QUESTIONABLE_THRESHOLD = 0.9
NUDENET_EXPOSED_THRESHOLD = 0.90
NUDENET_COVERED_THRESHOLD = 0.90

NUDENET_EXPOSED_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
}

NUDENET_COVERED_CLASSES = {
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
}

BROWSER_PROCESSES = [
    "chrome.exe", "msedge.exe", "firefox.exe",
    "brave.exe", "opera.exe", "iexplore.exe",
]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def load_wd_tagger_onnx():
    repo = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
    try:
        model_path = hf_hub_download(repo_id=repo, filename="model.onnx")
        tags_path = hf_hub_download(repo_id=repo, filename="selected_tags.csv")
        session = InferenceSession(model_path)
        rating_indices = {}
        with open(tags_path, newline='', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                if row.get('category', '') == '9':
                    rating_indices[row['name']] = i
        return session, rating_indices
    except Exception as e:
        log.warning(f"Failed to load WD Tagger ONNX: {e}")
        return None, {}


_WD_ZERO = {'explicit': 0.0, 'questionable': 0.0, 'sensitive': 0.0, 'general': 0.0}


def get_wd_rating(onnx_session, rating_indices: dict, img: Image.Image) -> dict:
    if onnx_session is None or not rating_indices:
        return _WD_ZERO
    try:
        bg = Image.new("RGBA", img.size, "WHITE")
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA"))
        arr = np.array(bg.convert("RGB").resize((448, 448), Image.LANCZOS), dtype=np.float32)[:, :, ::-1]
        arr = np.expand_dims(arr, 0)
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        scores = onnx_session.run([output_name], {input_name: arr})[0][0]
        return {label: float(scores[idx]) for label, idx in rating_indices.items()}
    except Exception as e:
        log.debug(f"WD Tagger error: {e}")
        return _WD_ZERO


def take_screenshot() -> Image.Image:
    with mss.MSS() as sct:
        shot = sct.grab(sct.monitors[1])
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def enforce(logoff: bool = False) -> None:
    for proc in BROWSER_PROCESSES:
        subprocess.run(["taskkill", "/F", "/IM", proc], capture_output=True)
    log.warning("Browsers closed.")
    if logoff:
        log.warning("Logging out user.")
        subprocess.run(["shutdown", "/l", "/f"], capture_output=True)


def _falconsai_score(nsfw_clf, img: Image.Image) -> float:
    img = img.convert("RGB")
    if img.width < 320 or img.height < 320:
        scale = max(320 / img.width, 320 / img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    results = nsfw_clf(img)
    return next((r["score"] for r in results if r["label"] == "nsfw"), 0.0)


def run_detection(nsfw_clf, wd_session, rating_indices, detector, img: Image.Image) -> tuple[bool, dict]:
    scores = {}

    scores['falconsai'] = _falconsai_score(nsfw_clf, img)

    rating = get_wd_rating(wd_session, rating_indices, img)
    scores['wd14_explicit'] = rating.get('explicit', 0.0)
    scores['wd14_questionable'] = rating.get('questionable', 0.0)

    if scores['wd14_explicit'] >= ANIME_EXPLICIT_THRESHOLD:
        return True, scores
    if scores['wd14_questionable'] >= ANIME_QUESTIONABLE_THRESHOLD:
        return True, scores
    if scores['falconsai'] >= FALCONSAI_THRESHOLD:
        return True, scores

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        img.save(tmp_path, format="PNG")
        nudenet_results = detector.detect(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    for d in nudenet_results:
        if d["class"] in NUDENET_EXPOSED_CLASSES and d["score"] >= NUDENET_EXPOSED_THRESHOLD:
            scores['nudenet'] = f"{d['class']}:{d['score']:.2f}"
            return True, scores
        if d["class"] in NUDENET_COVERED_CLASSES and d["score"] >= NUDENET_COVERED_THRESHOLD:
            scores['nudenet'] = f"{d['class']}:{d['score']:.2f}"
            return True, scores

    return False, scores


MAX_LOG_ENTRIES = 10


def log_detection(scores: dict) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        timestamp,
        f"falconsai={scores['falconsai']:.2f}",
        f"wd14_explicit={scores['wd14_explicit']:.2f}",
        f"wd14_questionable={scores['wd14_questionable']:.2f}",
    ]
    if 'nudenet' in scores:
        parts.append(f"nudenet={scores['nudenet']}")
    line = " | ".join(parts)
    DETECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    existing = DETECTION_LOG.read_text(encoding="utf-8").splitlines() if DETECTION_LOG.exists() else []
    entries = (existing + [line])[-MAX_LOG_ENTRIES:]
    DETECTION_LOG.write_text("\n".join(entries) + "\n", encoding="utf-8")
    log.warning(f"Flagged: {line}")


def _is_explicit(scores: dict) -> bool:
    if scores.get('wd14_explicit', 0.0) >= ANIME_EXPLICIT_THRESHOLD:
        return True
    nudenet = scores.get('nudenet', '')
    return any(cls in nudenet for cls in NUDENET_EXPOSED_CLASSES)


def save_flagged_screenshot(img: Image.Image) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"flagged_{timestamp}.png"
    img.save(path, format="PNG")
    log.warning(f"Screenshot saved: {path}")


def check_nudity(nsfw_clf, wd_session, rating_indices, detector, img: Image.Image) -> None:
    flagged, scores = run_detection(nsfw_clf, wd_session, rating_indices, detector, img)
    if flagged:
        log_detection(scores)
        save_flagged_screenshot(img)
        enforce(logoff=_is_explicit(scores))


def log_crash(exc: BaseException) -> None:
    CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}\n"
    with CRASH_LOG.open("a", encoding="utf-8") as f:
        f.write(entry)
    log.error(f"Crash logged: {type(exc).__name__}: {exc}")


def main() -> None:
    nsfw_clf = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    wd_session, rating_indices = load_wd_tagger_onnx()
    detector = NudeDetector()
    try:
        while True:
            try:
                img = take_screenshot()
                check_nudity(nsfw_clf, wd_session, rating_indices, detector, img)
            except mss.exception.ScreenShotError:
                pass
            time.sleep(SCREENSHOT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log_crash(exc)
        raise


if __name__ == "__main__":
    main()
