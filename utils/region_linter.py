"""
Region linter for UI Vision Bot

Checks regions for:
- zero or negative size
- out-of-bounds
- overlaps
- missing template images
- suspicious annotations

Designed to work with:
- ui_lab.py (visual overlays)
- main.py (offline processing)
- live_runner.py (live automation)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from PyQt6.QtCore import QRect


# =========================
# Region model
# =========================
@dataclass
class Region:
    name: str
    type: str                 # button | template | ocr | hybrid
    x: int
    y: int
    w: int
    h: int
    annotation: str = ""
    template_image: str = ""


# =========================
# Lint regions
# =========================
def lint_regions(
    regions: List[Region],
    img_w: int,
    img_h: int,
    run_dir: Path | None = None
) -> Dict[str, List[str]]:
    """
    Lint a list of regions.

    Returns:
        { region_name: [issues...] }
    """

    issues: Dict[str, List[str]] = {}

    def add(r: Region, msg: str):
        issues.setdefault(r.name, []).append(msg)

    # -------------------------
    # Size & bounds checks
    # -------------------------
    for r in regions:
        if r.w <= 0 or r.h <= 0:
            add(r, "zero or negative size")

        if r.x < 0 or r.y < 0:
            add(r, "negative coordinates")

        if r.x + r.w > img_w or r.y + r.h > img_h:
            add(r, "out of bounds")

        # Annotation sanity
        if r.annotation and len(r.annotation) > 64:
            add(r, f"annotation too long ({len(r.annotation)} chars)")

        # Template existence
        if r.type in ("template", "hybrid"):
            if not r.template_image:
                add(r, "template region missing template_image")
            elif run_dir:
                template_path = run_dir / r.template_image
                if not template_path.exists():
                    add(r, f"template image missing: {r.template_image}")

    # -------------------------
    # Overlap detection
    # -------------------------
    for i, a in enumerate(regions):
        rect_a = QRect(a.x, a.y, a.w, a.h)
        for b in regions[i + 1:]:
            rect_b = QRect(b.x, b.y, b.w, b.h)
            if rect_a.intersects(rect_b):
                add(a, f"overlaps {b.name}")
                add(b, f"overlaps {a.name}")

    return issues


# =========================
# YAML helper (optional)
# =========================
def regions_from_yaml(path: Path) -> List[Region]:
    """
    Load regions.yaml into Region objects.
    """
    import yaml

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
# CLI test utility
# =========================
if __name__ == "__main__":
    import cv2
    import sys

    if len(sys.argv) < 3:
        print("Usage: python region_linter.py <frame.png> <regions.yaml>")
        sys.exit(1)

    frame_path = Path(sys.argv[1])
    regions_path = Path(sys.argv[2])
    run_dir = regions_path.parent

    img = cv2.imread(str(frame_path))
    if img is None:
        print("❌ Frame image not found")
        sys.exit(1)

    h, w = img.shape[:2]
    regions = regions_from_yaml(regions_path)

    issues = lint_regions(regions, w, h, run_dir)

    if not issues:
        print("✓ No region issues found")
    else:
        print("⚠ Region issues:")
        for name, msgs in issues.items():
            for msg in msgs:
                print(f"  {name}: {msg}")
