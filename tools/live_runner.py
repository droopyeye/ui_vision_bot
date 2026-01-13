import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui

from utils.policy_engine import PolicyEngine
from main import (
    load_regions_yaml,
    load_policy_yaml,
    analyze_frame,
)

# -------------------------------
# CONFIG
# -------------------------------

EMERGENCY_STOP_KEY = "esc"
FRAME_DELAY = 0.1  # seconds between frames
DEFAULT_CONFIDENCE = 0.0


# -------------------------------
# LIVE RUNNER
# -------------------------------

class LiveRunner:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.running = False

        # Load config
        self.regions = load_regions_yaml(self.run_dir / "regions.yaml")
        self.policies = load_policy_yaml(self.run_dir / "policy.yaml")

        self.policy_engine = PolicyEngine(self.policies)

        # Safety
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05

        print(f"✅ Loaded {len(self.regions)} regions")
        print(f"✅ Loaded {len(self.policies)} policies")

    # ---------------------------
    # FRAME CAPTURE
    # ---------------------------

    def capture_frame(self) -> np.ndarray:
        """
        Capture full screen frame using OpenCV (via pyautogui).
        """
        img = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return frame

    # ---------------------------
    # CLICK LOGIC
    # ---------------------------

    def _click(self, region: dict, confidence: float):
        rect = region["rect"]
        x, y, w, h = rect

        mode = region.get("click", {}).get("mode", "center")
        offset = region.get("click", {}).get("offset", [0, 0])

        if mode == "center":
            cx = x + w // 2
            cy = y + h // 2
        else:
            cx = x
            cy = y

        cx += offset[0]
        cy += offset[1]

        print(
            f"🖱 Click {region['name']} @ ({cx}, {cy}) "
            f"conf={confidence:.2f}"
        )

        pyautogui.moveTo(cx, cy, duration=0.05)
        pyautogui.click()

    # ---------------------------
    # MAIN LOOP
    # ---------------------------

    def run(self):
        print("▶ Live runner started (ESC to stop)")
        self.running = True

        while self.running:
            # ---- Emergency stop ----
            if pyautogui.keyDown(EMERGENCY_STOP_KEY):
                print("🛑 Emergency stop key pressed")
                break

            # ---- Capture ----
            frame = self.capture_frame()

            # ---- Analyze ----
            analysis = analyze_frame(
                frame,
                self.regions,
                gpu=True,  # OCR GPU only (as requested)
            )

            # ---- Policy evaluation ----
            decision = self.policy_engine.evaluate(analysis)

            if decision:
                action = decision["action"]
                region_name = decision["region"]

                region = next(
                    r for r in self.regions if r["name"] == region_name
                )

                if action["type"] == "click":
                    self._click(region, decision.get("confidence", 0.0))

                elif action["type"] == "stop":
                    print("🛑 Policy stop triggered")
                    break

            time.sleep(FRAME_DELAY)

        self.running = False
        print("■ Live runner stopped")


# -------------------------------
# CLI ENTRY
# -------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="UI Vision Live Runner")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to debug run directory",
    )

    args = parser.parse_args()

    runner = LiveRunner(Path(args.run_dir))
    runner.run()


if __name__ == "__main__":
    main()
