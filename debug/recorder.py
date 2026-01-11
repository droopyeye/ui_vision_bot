from pathlib import Path
from datetime import datetime
import time
import cv2
import numpy as np
import mss
import json

class FrameRecorder:
    def __init__(self, root="debug_runs", fps=5, monitor=2):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / f"run_{ts}"
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.meta = {
            "start_time": ts,
            "fps": fps,
            "monitor": monitor,
        }

        self.fps = fps
        self.monitor = monitor
        self.idx = 0
        self.sct = mss.mss()

    def capture_frame(self):
        monitor = self.sct.monitors[self.monitor]
        img = np.array(self.sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        self.idx += 1
        path = self.frames_dir / f"{self.idx:06d}.png"
        cv2.imwrite(str(path), img)

    def run(self, duration=None):
        start = time.time()
        interval = 1.0 / self.fps

        try:
            while True:
                self.capture_frame()
                time.sleep(interval)

                if duration and (time.time() - start) > duration:
                    break
        finally:
            self.meta["frames"] = self.idx
            with open(self.run_dir / "meta.json", "w") as f:
                json.dump(self.meta, f, indent=2)
