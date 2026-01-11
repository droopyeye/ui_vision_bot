import easyocr

class OCRReader:
    def __init__(self, languages=["en"]):
        self.reader = easyocr.Reader(languages, gpu=False)

    def read(self, img):
        results = self.reader.readtext(img)
        parsed = []

        for box, text, conf in results:
            parsed.append({
                "text": text.lower(),
                "confidence": conf,
                "box": box
            })

        return parsed
