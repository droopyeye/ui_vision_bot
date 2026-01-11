import mss
import numpy as np
import cv2

class ScreenCapture:
    def __init__(self, monitor=1):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor]

    def grab(self):
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def grab_region(self, region):
        monitor = {
            "top": region["y"],
            "left": region["x"],
            "width": region["w"],
            "height": region["h"]
        }
        img = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
