from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import easyocr

from utils.region_linter import lint_regions
from utils.hybrid_eval import aggregate_confidence


# -----------------------------
# OCR Reader (lazy init)
# -----------------------------

_OCR_READER = None


def get_ocr_reader(gpu=True):
    global _OCR_READER
    if _OCR_READER is None:
        _OCR_READER = easyocr.Reader(["en"], gpu=gpu)
    return _OCR_READER


# -----------------------------
# Region loading
# -----------------------------

def load_regions_yaml(path: Path) -> List[dict]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        regions = yaml.safe_load(f)

    if not isinstance(regions, list):
        raise ValueError("regions.yaml must contain a list")

    return regions

# -----------------------------
# Policy loading
# -----------------------------
def load_policy_yaml(path: Path) -> list[dict]:
    import yaml

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("policies", [])


# -----------------------------
# Template matching
# -----------------------------

def match_templates(
    frame: np.ndarray,
    regions: List[dict],
) -> Dict[str, Tuple[bool, float]]:
    """
    Returns:
        { region_name: (matched, confidence) }
    """
    results = {}

    for r in regions:
        name = r["name"]
        if r["type"] not in {"template", "hybrid"}:
            continue

        tmpl_cfg = r.get("template")
        if not tmpl_cfg:
            results[name] = (False, 0.0)
            continue

        rect = r["rect"]
        x, y, w, h = rect
        roi = frame[y : y + h, x : x + w]

        tmpl_path = Path(tmpl_cfg["image"])
        if not tmpl_path.exists():
            results[name] = (False, 0.0)
            continue

        tmpl = cv2.imread(str(tmpl_path))
        if tmpl is None:
            results[name] = (False, 0.0)
            continue

        method = tmpl_cfg.get("method", cv2.TM_CCOEFF_NORMED)
        res = cv2.matchTemplate(roi, tmpl, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        threshold = tmpl_cfg.get("threshold", 0.8)
        results[name] = (max_val >= threshold, float(max_val))

    return results


# -----------------------------
# OCR
# -----------------------------

def run_ocr(
    frame: np.ndarray,
    regions: List[dict],
    gpu=True,
) -> Dict[str, Tuple[bool, float]]:
    """
    Returns:
        { region_name: (matched, confidence) }
    """
    reader = get_ocr_reader(gpu=gpu)
    results = {}

    for r in regions:
        name = r["name"]
        if r["type"] not in {"ocr", "hybrid"}:
            continue

        ocr_cfg = r.get("ocr")
        if not ocr_cfg:
            results[name] = (False, 0.0)
            continue

        rect = r["rect"]
        x, y, w, h = rect
        roi = frame[y : y + h, x : x + w]

        detections = reader.readtext(roi)

        target = ocr_cfg.get("text", "")
        match_mode = ocr_cfg.get("match", "contains")
        min_conf = ocr_cfg.get("confidence", 0.5)

        best_conf = 0.0
        matched = False

        for _, text, conf in detections:
            norm_text = text.strip()
            if not norm_text:
                continue

            text_match = False
            if match_mode == "exact":
                text_match = norm_text == target
            elif match_mode == "contains":
                text_match = target.lower() in norm_text.lower()
            elif match_mode == "regex":
                import re
                text_match = bool(re.search(target, norm_text))

            if text_match:
                best_conf = max(best_conf, conf)
                matched = conf >= min_conf

        results[name] = (matched, float(best_conf))

    return results


# -----------------------------
# Hybrid evaluation (shared)
# -----------------------------

def evaluate_hybrid_region(region, template_result, ocr_result):
    logic = region.get("logic", {})
    require = logic.get("require", ["template", "ocr"])
    aggregate = logic.get("aggregate", "min")

    checks = {
        "template": template_result,
        "ocr": ocr_result,
    }

    for key in require:
        matched, _ = checks[key]
        if not matched:
            return False, 0.0, key

    confidences = [checks[k][1] for k in require]
    return True, aggregate_confidence(confidences, aggregate), None


# -----------------------------
# Frame analysis entry point
# -----------------------------

def analyze_frame(
    frame: np.ndarray,
    regions: List[dict],
    gpu=True,
):
    """
    Unified analysis used by UI Lab, replay viewer, or live runner.

    Returns:
        {
          region_name: {
            matched: bool,
            confidence: float,
            type: str
          }
        }
    """
    template_results = match_templates(frame, regions)
    ocr_results = run_ocr(frame, regions, gpu=gpu)

    results = {}

    for r in regions:
        name = r["name"]
        rtype = r["type"]

        if rtype == "template":
            m, c = template_results.get(name, (False, 0.0))
            results[name] = {
                "matched": m,
                "confidence": c,
                "type": rtype,
            }

        elif rtype == "ocr":
            m, c = ocr_results.get(name, (False, 0.0))
            results[name] = {
                "matched": m,
                "confidence": c,
                "type": rtype,
            }

        elif rtype == "hybrid":
            tmpl = template_results.get(name, (False, 0.0))
            ocr = ocr_results.get(name, (False, 0.0))
            ok, conf, failed = evaluate_hybrid_region(r, tmpl, ocr)
            results[name] = {
                "matched": ok,
                "confidence": conf,
                "type": rtype,
                "failed": failed,
            }

    return results
