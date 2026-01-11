class DetectionFusion:
    def validate_ocr(self, ocr_results, expected, threshold):
        for item in ocr_results:
            if item["confidence"] < threshold:
                continue
            for word in expected:
                if word in item["text"]:
                    return True, item
        return False, None

    def fuse(self, img_match, ocr_ok):
        if not img_match["found"]:
            return False

        return img_match["confidence"] > 0.9 or ocr_ok
