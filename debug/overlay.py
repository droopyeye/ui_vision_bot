import cv2
import numpy as np

class DebugOverlay:
    def draw_region(self, frame, region, color=(255, 0, 0)):
        x, y, w, h = region.values()
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    def draw_template_result(self, frame, region, match):
        if not match["found"]:
            return

        x = region["x"] + match["location"][0]
        y = region["y"] + match["location"][1]

        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{match['confidence']:.2f}",
            (x+10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    def draw_ocr(self, frame, region, ocr_results):
        for item in ocr_results:
            pts = [(int(x), int(y)) for x, y in item["box"]]
            cv2.polylines(frame, [np.array(pts)], True, (0, 255, 255), 2)
