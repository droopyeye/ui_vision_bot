import yaml
import cv2
import sys
import os
from utils.region_linter import lint_regions
from capture.screen_capture import ScreenCapture
from vision.matcher import TemplateMatcher
from vision.ocr import OCRReader
from vision.fusion import DetectionFusion
from debug.overlay import DebugOverlay

config = yaml.safe_load(open("config/regions.yaml"))
errors = lint_regions(config)
if errors:
    print("\nCONFIGURATION ERRORS:")
    for e in errors:
        print(" -", e)
    sys.exit(1)

cap = ScreenCapture()
matcher = TemplateMatcher()
ocr = OCRReader()
fusion = DetectionFusion()
overlay = DebugOverlay()


def validate_config(regions):
    for name, cfg in regions.items():
        if "template" in cfg:
            path = f"config/templates/{cfg['template']}"
            if cfg["template"] and not os.path.exists(path):
                print(f"[CONFIG ERROR] Missing template for {name}: {path}")


templates = {}

for name, r in config["regions"].items():
    if r["mode"] not in ("template", "hybrid"):
        continue

    path = f"config/templates/{r['template']['file']}"
    templates[name] = cv2.imread(path)


while True:
    frame = cap.grab()

    for name, region in config["regions"].items():
        r = region
        mode = r["mode"]
        region_img = cap.grab_region(region)
                
        if mode == "template":
            match = matcher.match(
                region_img,
                templates[name],
                region["template_threshold"]
            )
            final = match["found"]

        elif mode == "ocr":
            ocr_results = ocr.read(region_img)
            ocr_ok, _ = fusion.validate_ocr(
                ocr_results,
                region["ocr"]["expected"],
                region["ocr"]["confidence_threshold"]
            )
            final = ocr_ok
        
        elif mode == "hybrid":
            match = matcher.match(
                region_img,
                templates[name],
                region["template_threshold"]
            )
            ocr_results = ocr.read(region_img)
            ocr_ok, _ = fusion.validate_ocr(
                ocr_results,
                region["ocr"]["expected"],
                region["ocr"]["confidence_threshold"]
            )
            final = fusion.fuse(match, ocr_ok)
        
        elif mode == "yolo":
            # YOLO detection logic would go here
            final = False   
        
        else:
            match = {"found": False, "confidence": 0.0}
            ocr_ok = False


        final = fusion.fuse(match, ocr_ok)

        overlay.draw_region(frame, region)
        overlay.draw_template_result(frame, region, match)

    cv2.imshow("UI Vision Debug", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
