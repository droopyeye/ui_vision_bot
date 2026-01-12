"""
main.py

Offline runner for UI Vision Bot.
Processes frames from a run_dir using regions.yaml,
runs OCR and template matching, and prints results.

Safe to import from live_runner.py (no side effects).
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import cv2
import numpy as np
import yaml
import easyocr

from utils.region_linter import lint_regions, Region


# =========================
# Regions loader
# =========================
def load_regions_yaml(path: Path) -> List[Region]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []

    regions: List[Region] = []
    for d in data:
        regions.append(Region(
            name=d.get("name", ""),
            type=d.get("type", "button"),
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            w=int(d.get("w", 0)),
            h=int(d.get("h", 0)),
            annotation=d.get("annotation", ""),
            template_image=d.get("template_image", "")
        ))
    return regions


# =========================
# OCR
# =========================
def run_ocr(frame: np.ndarray, regions: List[Region], reader) -> Dict[str, str]:
    results = {}

    for r in regions:
        if r.type not in ("ocr", "hybrid"):
            continue

        roi = frame[r.y:r.y + r.h, r.x:r.x + r.w]
        if roi.size == 0:
            results[r.name] = ""
            continue

        try:
            text = reader.readtext(roi, detail=0)
            results[r.name] = " ".join(text)
        except Exception:
            results[r.name] = ""

    return results


# =========================
# Template matching
# =========================
def match_templates(
    frame: np.ndarray,
    regions: List[Region],
    run_dir: Path
) -> Dict[str, tuple]:
    matches = {}

    for r in regions:
        if r.type not in ("template", "hybrid"):
            continue
        if not r.template_image:
            continue

        template_path = run_dir / r.template_image
        if not template_path.exists():
            continue

        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            continue

        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        matches[r.name] = (max_val, max_loc)

    return matches


# =========================
# Frame loader
# =========================
def load_frames(run_dir: Path) -> List[Path]:
    frame_dir = run_dir / "frames"
    if not frame_dir.exists():
        return []
    return sorted(frame_dir.glob("*.png"))


# =========================
# Main execution
# =========================
def main(run_dir_path: str):
    run_dir = Path(run_dir_path)
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return

    frames = load_frames(run_dir)
    if not frames:
        print("❌ No frames found")
        return

    regions = load_regions_yaml(run_dir / "regions.yaml")
    if not regions:
        print("⚠ No regions defined")

    # Load first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        print("❌ Failed to load first frame")
        return

    h, w = first_frame.shape[:2]

    # Lint regions (SAFE HERE)
    issues = lint_regions(regions, w, h, run_dir)
    if issues:
        print("⚠ Region lint issues:")
        for name, msgs in issues.items():
            for msg in msgs:
                print(f"  {name}: {msg}")

    # OCR reader (CPU-safe for AMD GPU systems)
    reader = easyocr.Reader(["en"], gpu=False)

    # Process frames
    for idx, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        ocr_results = run_ocr(frame, regions, reader)
        template_results = match_templates(frame, regions, run_dir)

        print(f"\nFrame {idx + 1}/{len(frames)}: {frame_path.name}")

        for r in regions:
            outputs = []

            if r.type in ("ocr", "hybrid"):
                outputs.append(f"OCR='{ocr_results.get(r.name, '')}'")

            if r.type in ("template", "hybrid"):
                match = template_results.get(r.name)
                if match:
                    outputs.append(f"match={match[0]:.2f}")

            if outputs:
                print(f"  - {r.name} [{r.type}] ({r.annotation}): " + " | ".join(outputs))


# =========================
# CLI entry point
# =========================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <run_dir>")
        sys.exit(1)

    main(sys.argv[1])
