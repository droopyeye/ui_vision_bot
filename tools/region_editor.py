import cv2
import yaml
import numpy as np
import easyocr
from capture.screen_capture import ScreenCapture

REGION_MODES = {
    ord("1"): "template",
    ord("2"): "ocr",
    ord("3"): "hybrid",
    ord("4"): "yolo",
}

class RegionEditor:
    def __init__(self):
        self.cap = ScreenCapture()
        self.frame = self.cap.grab()
        self.clone = self.frame.copy()

        self.start = None
        self.end = None
        self.drawing = False

        self.regions = {}
        self.current_mode = "template"

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.frame = self.clone.copy()
            cv2.rectangle(self.frame, self.start, (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.end = (x, y)
            self.drawing = False
            cv2.rectangle(self.frame, self.start, self.end, (0, 255, 0), 2)

    def add_region(self):
        name = input("Region name: ").strip()
        if not name:
            print("Invalid name")
            return

        x1, y1 = self.start
        x2, y2 = self.end

        rect = {
            "x": min(x1, x2),
            "y": min(y1, y2),
            "w": abs(x2 - x1),
            "h": abs(y2 - y1),
        }

        region = {
            "mode": self.current_mode,
            "rect": rect,
        }

        if self.current_mode in ("template", "hybrid"):
            fname = f"{name}.png"
            crop = self.clone[
                rect["y"]:rect["y"] + rect["h"],
                rect["x"]:rect["x"] + rect["w"]
            ]
            cv2.imwrite(f"config/templates/{fname}", crop)
            region["template"] = {
                "file": fname,
                "threshold": 0.8,
            }

        if self.current_mode in ("ocr", "hybrid"):
            region["ocr"] = {
                "expected": [],
                "confidence_threshold": 0.6,
            }

        if self.current_mode == "yolo":
            region.pop("rect", None)
            region["class"] = input("YOLO class (e.g. button): ").strip()

        self.regions[name] = region
        print(f"Added region '{name}' ({self.current_mode})")

    def run(self):
        cv2.namedWindow("Region Editor")
        cv2.setMouseCallback("Region Editor", self.mouse_cb)

        while True:
            cv2.imshow("Region Editor", self.frame)
            key = cv2.waitKey(1) & 0xFF

            if key in REGION_MODES:
                self.current_mode = REGION_MODES[key]
                print(f"Mode set to {self.current_mode}")

            elif key == 13:  # Enter
                if self.start and self.end:
                    self.add_region()
                    self.clone = self.frame.copy()

            elif key == ord("s"):
                with open("config/regions.yaml", "w") as f:
                    yaml.dump({"regions": self.regions}, f)
                print("Saved regions.yaml")

            elif key == ord("q"):
                break

        cv2.destroyAllWindows()
