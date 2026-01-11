import cv2
import numpy as np

class TemplateMatcher:
    def match(self, region_img, template_img, threshold):
        res = cv2.matchTemplate(
            region_img, template_img, cv2.TM_CCOEFF_NORMED
        )

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        return {
            "found": max_val >= threshold,
            "confidence": float(max_val),
            "location": max_loc,
            "heatmap": res
        }
