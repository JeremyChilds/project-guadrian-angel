import logging
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import mss
from nudenet import NudeDetector
from PIL import Image
from transformers import pipeline
from huggingface_hub import hf_hub_download
import numpy as np
from onnxruntime import InferenceSession

SCREENSHOT_INTERVAL_SECONDS = 8
RETENTION_DAYS = 7
LOGS_DIR = Path(__file__).parent / "logs"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
FILENAME_PREFIX = "screenshot_"

FALCONSAI_THRESHOLD = 0.40
ANIME_EXPLICIT_THRESHOLD = 0.10
ANIME_QUESTIONABLE_THRESHOLD = 0.15
NUDENET_EXPOSED_THRESHOLD = 0.45
NUDENET_COVERED_THRESHOLD = 0.70

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
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def load_wd_tagger_onnx():
    """Load WD14 ONNX model from Hugging Face."""
    try:
        model_path = hf_hub_download(
            repo_id="SmilingWolf/wd-v1-4-vit-tagger",
            filename="model.onnx"
        )
        session = InferenceSession(model_path)
        log.info("WD Tagger ONNX model loaded successfully")
        return session
    except Exception as e:
        log.warning(f"Failed to load WD Tagger ONNX: {e}")
        return None


def get_wd_rating(onnx_session, image_path: Path) -> dict:
    """Get WD14 ratings from ONNX model."""
    if onnx_session is None:
        return {'explicit': 0.0, 'questionable': 0.0, 'sensitive': 0.0, 'general': 0.0}
    
    try:
        img = Image.open(str(image_path)).convert('RGB')
        # Upscale if too small (improves detection of zoomed-out content)
        if img.width < 448 or img.height < 448:
            scale_factor = max(448 / img.width, 448 / img.height)
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.LANCZOS)
        # Resize to 448x448 (WD14 input size)
        img = img.resize((448, 448), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        # Normalize
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img_array = (img_array - mean) / std
        img_array = np.transpose(img_array, (2, 0, 1)).astype(np.float32)
        img_array = np.expand_dims(img_array, 0).astype(np.float32)
        
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        outputs = onnx_session.run([output_name], {input_name: img_array})
        
        # WD14 outputs: general, sensitive, questionable, explicit
        scores = outputs[0][0]
        return {
            'general': float(scores[0]),
            'sensitive': float(scores[1]),
            'questionable': float(scores[2]),
            'explicit': float(scores[3])
        }
    except Exception as e:
        log.debug(f"Error processing with WD Tagger: {e}")
        return {'explicit': 0.0, 'questionable': 0.0, 'sensitive': 0.0, 'general': 0.0}


def ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def take_screenshot() -> Path:
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    filename = LOGS_DIR / f"{FILENAME_PREFIX}{timestamp}.png"
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        img.save(filename, format="PNG")
    log.info(f"Saved: {filename.name}")
    return filename


def enforce() -> None:
    for proc in BROWSER_PROCESSES:
        subprocess.run(["taskkill", "/F", "/IM", proc], capture_output=True)
    log.warning("Browsers closed.")
    # subprocess.run(["shutdown", "/l", "/f"])


def _falconsai_score(nsfw_clf, image_path: Path) -> float:
    img = Image.open(str(image_path)).convert('RGB')
    # Upscale if too small
    if img.width < 320 or img.height < 320:
        scale_factor = max(320 / img.width, 320 / img.height)
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
    # Save upscaled image to temp path for processing
    temp_path = image_path.parent / f"temp_{image_path.name}"
    img.save(temp_path)
    results = nsfw_clf(str(temp_path))
    temp_path.unlink()  # Clean up temp file
    return next((r["score"] for r in results if r["label"] == "nsfw"), 0.0)


def is_flagged(nsfw_clf, wd_session, detector, image_path: Path) -> bool:
    fc = _falconsai_score(nsfw_clf, image_path)
    log.debug(f"Falconsai score={fc:.2f} for {image_path.name}")
    
    # Always check WD14 for logging
    if wd_session is not None:
        rating = get_wd_rating(wd_session, image_path)
        log.debug(f"WD14 scores for {image_path.name}: explicit={rating['explicit']:.2f}, questionable={rating['questionable']:.2f}")
        if rating["explicit"] >= ANIME_EXPLICIT_THRESHOLD:
            log.warning(f"WD14 explicit score={rating['explicit']:.2f} in {image_path.name}")
            return True
        if rating["questionable"] >= ANIME_QUESTIONABLE_THRESHOLD:
            log.warning(f"WD14 questionable score={rating['questionable']:.2f} in {image_path.name}")
            return True
    
    if fc >= FALCONSAI_THRESHOLD:
        log.warning(f"Falconsai score={fc:.2f} in {image_path.name}")
        return True

    nudenet_results = detector.detect(str(image_path))
    for d in nudenet_results:
        if d["class"] in NUDENET_EXPOSED_CLASSES and d["score"] >= NUDENET_EXPOSED_THRESHOLD:
            log.warning(f"NudeNet exposed {d['class']} score={d['score']:.2f} in {image_path.name}")
            return True
        if d["class"] in NUDENET_COVERED_CLASSES and d["score"] >= NUDENET_COVERED_THRESHOLD:
            log.warning(f"NudeNet covered {d['class']} score={d['score']:.2f} in {image_path.name}")
            return True

    return False


def check_nudity(nsfw_clf, wd_session, detector, image_path: Path) -> None:
    if is_flagged(nsfw_clf, wd_session, detector, image_path):
        enforce()
    else:
        image_path.unlink()
        log.info("Clean — deleted screenshot")


def delete_old_screenshots() -> None:
    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
    deleted = 0
    for f in LOGS_DIR.glob(f"{FILENAME_PREFIX}*.png"):
        try:
            stem = f.stem.removeprefix(FILENAME_PREFIX)
            if datetime.strptime(stem, TIMESTAMP_FORMAT) < cutoff:
                f.unlink()
                deleted += 1
        except (ValueError, OSError):
            pass
    if deleted:
        log.info(f"Deleted {deleted} screenshot(s) older than {RETENTION_DAYS} day(s).")


def main() -> None:
    ensure_logs_dir()
    log.info("Loading Falconsai NSFW detector...")
    nsfw_clf = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    log.info("Loading WD Tagger ONNX model...")
    wd_session = load_wd_tagger_onnx()
    log.info("Loading NudeNet detector...")
    detector = NudeDetector()
    log.info(
        f"Guardian Angel started — interval={SCREENSHOT_INTERVAL_SECONDS}s, "
        f"retention={RETENTION_DAYS}d, saving to {LOGS_DIR}"
    )
    log.info("Press Ctrl+C to stop.")
    try:
        while True:
            path = take_screenshot()
            check_nudity(nsfw_clf, wd_session, detector, path)
            delete_old_screenshots()
            time.sleep(SCREENSHOT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        log.info("Guardian Angel stopped.")


if __name__ == "__main__":
    main()
