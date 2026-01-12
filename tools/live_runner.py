import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import mss
import yaml
import easyocr
import pyautogui
import keyboard

from utils.region_linter import lint_regions, Region
from main import run_ocr, match_templates, load_regions_yaml


# =========================
# Configuration
# =========================
CAPTURE_INTERVAL = 0.5   # seconds
CLICK_COOLDOWN = 1.0     # seconds
EMERGENCY_STOP_KEY = "esc"


# =========================
# Action model
# =========================
@dataclass
class Action:
    type: str           # "click"
    target: str         # region name


# =========================
# Live Vision Engine
# =========================
class LiveVisionEngine:
    def __init__(self, regions, reader, run_dir):
        self.regions = regions
        self.reader = reader
        self.run_dir = run_dir

    def process(self, frame):
        ocr = run_ocr(frame, self.regions, self.reader)
        templates = match_templates(frame, self.regions, self.run_dir)
        return {
            "ocr": ocr,
            "templates": templates
        }


# =========================
# State Resolver (minimal)
# =========================
class StateResolver:
    def __init__(self, regions):
        self.regions = regions

    def resolve(self, vision) -> str:
        # Example: main menu if start button visible
        for r in self.regions:
            if r.type in ["template", "hybrid"]:
                match = vision["templates"].get(r.name)
                if match and match[0] > 0.85:
                    return "main_menu"
        return "unknown"


# =========================
# Action Engine
# =========================
class ActionEngine:
    def __init__(self):
        self.last_action = 0.0

    def can_act(self):
        return time.time() - self.last_action > CLICK_COOLDOWN

    def execute(self, action: Action, regions):
        if not self.can_act():
            return

        region = next((r for r in regions if r.name == action.target), None)
        if not region:
            return

        x = region.x + region.w // 2
        y = region.y + region.h // 2

        if action.type == "click":
            pyautogui.moveTo(x, y, duration=0.1)
            pyautogui.click()
            self.last_action = time.time()
            print(f"[ACTION] Clicked {region.name} at ({x},{y})")


# =========================
# Live Runner
# =========================
class LiveRunner:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.regions = load_regions_yaml(run_dir / "regions.yaml")
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.vision = LiveVisionEngine(self.regions, self.reader, run_dir)
        self.actions = ActionEngine()
        self.state_resolver = StateResolver(self.regions)

        self.running = True

        self._lint()

    def _lint(self):
        with mss.mss() as sct:
            monitor = sct.monitors[2]
            img = np.array(sct.grab(monitor))
            h, w = img.shape[:2]
        issues = lint_regions(self.regions, w, h, self.run_dir)
        if issues:
            print("⚠ Region lint issues:")
            for k, msgs in issues.items():
                for m in msgs:
                    print(f"  {k}: {m}")

    def capture_frame(self):
        with mss.mss() as sct:
            monitor = sct.monitors[2]
            img = np.array(sct.grab(monitor))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def run(self):
        print("▶ Live runner started (ESC to stop)")
        last_state = None

        while self.running:
            frame = self.capture_frame()
            vision = self.vision.process(frame)
            state = self.state_resolver.resolve(vision)

            if state != last_state:
                print(f"[STATE] {state}")
                last_state = state

            # Example policy
            if state == "main_menu":
                self.actions.execute(Action("click", "start_button"), self.regions)

            if keyboard.is_pressed(EMERGENCY_STOP_KEY):
                print("Emergency stop pressed!")
                self.running = False
                break

            time.sleep(CAPTURE_INTERVAL)


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python live_runner.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print("Run directory not found")
        sys.exit(1)

    runner = LiveRunner(run_dir)
    runner.run()
