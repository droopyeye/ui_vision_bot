# live_runner.py
import time
from pathlib import Path
import cv2
import numpy as np
import mss
import pyautogui
import yaml

from main import Region, analyze_region, draw_debug_overlay, reader

# -------------------------------
# Config
# -------------------------------
RUN_DIR = Path("debug_runs/run_latest")  # set to your run folder
DEBUG_OVERLAY = True
EMERGENCY_STOP_KEY = "esc"  # press to stop the runner
CLICK_ENABLED = False       # set True to execute clicks
MATCH_INTERVAL = 0.5        # seconds between frame analyses

# -------------------------------
# Load regions from YAML
# -------------------------------
def load_regions_yaml(run_dir: Path):
    yaml_file = run_dir / "regions.yaml"
    regions = []
    if yaml_file.exists():
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        for r in data:
            regions.append(
                Region(
                    name=r.get("name"),
                    rect=r.get("rect"),
                    type=r.get("type","template"),
                    template_image=r.get("template_image"),
                    click=r.get("click"),
                    annotation=r.get("annotation","")
                )
            )
    else:
        print(f"⚠️ No regions.yaml found in {run_dir}")
    return regions

regions = load_regions_yaml(RUN_DIR)
if not regions:
    print("No regions loaded. Exiting.")
    exit(1)

# -------------------------------
# Screen capture setup
# -------------------------------
sct = mss.mss()
monitor = sct.monitors[2]  # change monitor index if needed

# -------------------------------
# Live runner loop
# -------------------------------
try:
    while True:
        # Emergency stop
        if pyautogui.keyDown(EMERGENCY_STOP_KEY):
            print("Emergency stop pressed!")
            break

        # Capture screen
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Analyze each region
        for r in regions:
            analyze_region(frame, r, RUN_DIR, ocr_reader=reader)

            # Optional click execution
            if CLICK_ENABLED and r.matched and r.click:
                x, y, w, h = r.rect
                mode = r.click.get("mode", "center")
                offset = r.click.get("offset", [0,0])
                cx, cy = (x + w//2, y + h//2) if mode=="center" else (x, y)
                cx += offset[0]; cy += offset[1]
                pyautogui.click(cx, cy)
                print(f"Clicked {r.name} at {cx},{cy}")

        # Draw debug overlay
        if DEBUG_OVERLAY:
            overlay_frame = draw_debug_overlay(frame, regions)
            cv2.imshow("Live Debug Overlay", overlay_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Sleep to reduce CPU load
        time.sleep(MATCH_INTERVAL)

finally:
    cv2.destroyAllWindows()
