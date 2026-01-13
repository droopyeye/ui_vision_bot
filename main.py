# main.py
import cv2
from pathlib import Path
import numpy as np
import easyocr

# -------------------------------
# Region class definition
# -------------------------------
class Region:
    def __init__(self, name, rect, type="template", template_image=None, ocr_text="", click=None, annotation=""):
        self.name = name
        self.rect = rect  # [x, y, w, h]
        self.type = type  # template, ocr, hybrid
        self.template_image = template_image
        self.ocr_text = ocr_text
        self.click = click  # {"mode": "center", "offset": [0,0]}
        self.annotation = annotation

        # runtime confidence
        self.template_confidence = 0.0
        self.ocr_confidence = 0.0
        self.hybrid_confidence = 0.0
        self.matched = False

# -------------------------------
# OCR Reader
# -------------------------------
reader = easyocr.Reader(["en"], gpu=True)

# -------------------------------
# Template matching helper
# -------------------------------
def match_template_region(frame, region, run_dir):
    """Return template confidence for a single region."""
    if not region.template_image:
        return 0.0

    tmpl_path = (Path(run_dir) / region.template_image).resolve()
    tmpl = cv2.imread(str(tmpl_path), cv2.IMREAD_UNCHANGED)
    if tmpl is None:
        print(f"⚠️ Template not found for region {region.name}: {tmpl_path}")
        return 0.0

    x, y, w, h = region.rect
    roi = frame[y:y+h, x:x+w]

    # ROI must be large enough for template
    if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
        print(f"⚠️ ROI smaller than template for {region.name}")
        return 0.0

    # Grayscale for robustness
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(roi_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    return float(max_val)

# -------------------------------
# Analyze a single region
# -------------------------------
def analyze_region(frame, region, run_dir, ocr_reader=reader):
    """
    Compute template, OCR, and hybrid confidence for a region.
    Updates region.matched according to thresholds.
    """
    # Template confidence
    template_conf = match_template_region(frame, region, run_dir)

    # OCR confidence
    ocr_conf = 0.0
    if region.type in ["ocr", "hybrid"]:
        x, y, w, h = region.rect
        roi = frame[y:y+h, x:x+w]
        result = ocr_reader.readtext(roi)
        ocr_conf = max([conf for _, text, conf in result], default=0.0)

    # Hybrid
    hybrid_conf = 0.0
    if region.type == "hybrid":
        hybrid_conf = (template_conf + ocr_conf) / 2.0

    # Update region object
    region.template_confidence = template_conf
    region.ocr_confidence = ocr_conf
    region.hybrid_confidence = hybrid_conf

    # Determine matched status
    threshold = 0.7
    if region.type == "hybrid":
        region.matched = hybrid_conf >= threshold
    elif region.type == "ocr":
        region.matched = ocr_conf >= threshold
    elif region.type == "template":
        region.matched = template_conf >= threshold

    return region.matched

# -------------------------------
# Run analysis on all regions
# -------------------------------
def analyze_frame(frame, regions, run_dir):
    for r in regions:
        analyze_region(frame, r, run_dir)
    return regions

# -------------------------------
# Debug overlay for visualization
# -------------------------------
def draw_debug_overlay(frame, regions):
    """
    Draw rectangles, click points, and confidence labels.
    Returns a new frame with overlays.
    """
    frame_overlay = frame.copy()
    for r in regions:
        x, y, w, h = r.rect
        color = (0,255,0) if r.matched else (0,0,255)
        cv2.rectangle(frame_overlay, (x,y), (x+w, y+h), color, 2)

        # Click point
        if r.click:
            mode = r.click.get("mode", "center")
            offset = r.click.get("offset", [0,0])
            cx, cy = (x + w//2, y + h//2) if mode=="center" else (x, y)
            cx += offset[0]; cy += offset[1]
            cv2.circle(frame_overlay, (cx, cy), 5, (255,0,0), -1)

        # Confidence label
        label = f"{r.name} | Tmpl:{r.template_confidence:.2f} OCR:{r.ocr_confidence:.2f}"
        if r.type=="hybrid":
            label += f" Hybrid:{r.hybrid_confidence:.2f}"
        cv2.putText(frame_overlay, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    return frame_overlay

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    run_dir = Path(".")  # change as needed
    frame_path = run_dir / "debug_runs/frame_example.png"
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    # Example regions
    regions = [
        Region(
            name="button_undock",
            rect=[330, 100, 100, 50],
            type="hybrid",
            template_image="templates/button-undock.png",
            click={"mode":"center","offset":[0,0]}
        )
    ]

    # Analyze frame
    analyze_frame(frame, regions, run_dir)

    # Overlay debug
    frame_overlay = draw_debug_overlay(frame, regions)
    cv2.imshow("Debug Overlay", frame_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
