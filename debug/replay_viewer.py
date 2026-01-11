import cv2
import json
import yaml
import numpy as np
from pathlib import Path

CLASS_MAP = {
    ord("1"): 0,  # button
    ord("2"): 1,  # icon
    ord("3"): 2,  # text_block
    ord("4"): 3,  # panel
    ord("5"): 4,  # dialog
    ord("6"): 6,  # checkbox
    ord("7"): 7,  # slider
    ord("8"): 8,  # dropdown
    ord("9"): 10, # progress_bar
    ord("0"): 11, # notification
}

class ReplayViewer:
    def __init__(self, run_dir, regions_config):
        self.run_dir = Path(run_dir)
        self.frames = sorted((self.run_dir / "frames").glob("*.png"))
        self.events = self._load_events()
        self.regions = regions_config["regions"]

        self.idx = 0
        self.playing = False
        self.selected_event_idx = 0
        self.labels_file = open(self.run_dir / "labels.jsonl", "a")

    def _load_events(self):
        events = {}
        with open(self.run_dir / "events.jsonl", "r") as f:
            for line in f:
                e = json.loads(line)
                events.setdefault(e["frame"], []).append(e)
        return events
    
    def _load_region_crop(self, region_name):
        path = (
        self.run_dir
        / "regions"
        / region_name
        / f"{self.idx:06d}.png"
        )
        if path.exists():
            img = cv2.imread(str(path))
        return cv2.resize(img, None, fx=2, fy=2)
        return None

    def _label_event(self, event, label):
        record = {
            "frame": self.idx,
            "region": event["region"],
            "label": label
        }
        self.labels_file.write(json.dumps(record) + "\n")
        self.labels_file.flush()

    def _draw_overlays(self, frame):
        frame_events = self.events.get(self.idx, [])

        for e in frame_events:
            region = self.regions[e["region"]]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            # Color code decision
            color = (0, 255, 0) if e["final_decision"] else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            label = f"{e['region']} | T:{e['template']['confidence']:.2f}"
            if e["ocr_valid"]:
                label += " OCR✓"

            cv2.putText(
                frame,
                label,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Draw template match point
            if e["template"]["found"]:
                tx, ty = e["template"]["location"]
                cx, cy = x + tx, y + ty
                cv2.circle(frame, (cx, cy), 6, (255, 255, 0), 2)

        return frame

    def _render_side_panel(self, event):
        panel = 255 * np.ones((400, 400, 3), dtype=np.uint8)

        y = 30
        def line(txt):
            nonlocal y
            cv2.putText(panel, txt, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            y += 24

        line(f"Region: {event['region']}")
        line(f"T conf: {event['template']['confidence']:.2f}")
        line(f"OCR valid: {event['ocr_valid']}")
        line(f"Decision: {event['final_decision']}")

        line("")
        line("[Y] True Positive")
        line("[N] False Positive")
        line("[U] Uncertain")
        line("[I] Ignore")

        return panel

    def _jump(self, direction=1):
        step = 1 if direction > 0 else -1
        i = self.idx + step
        while 0 <= i < len(self.frames):
            if any(e["final_decision"] for e in self.events.get(i, [])):
                self.idx = i
                return
            i += step

    def export_training_sample(self, event, class_id):
        out = self.run_dir / "training_export"
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(exist_ok=True)

        img = cv2.imread(str(self.frames[self.idx]))
        h, w, _ = img.shape

        img_name = f"frame_{self.idx:06d}.png"
        cv2.imwrite(str(out / "images" / img_name), img)

        r = self.regions[event["region"]]
        xc = (r["x"] + r["w"]/2) / w
        yc = (r["y"] + r["h"]/2) / h
        ww = r["w"] / w
        hh = r["h"] / h

        label = f"{class_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n"

        with open(out / "labels" / img_name.replace(".png", ".txt"), "w") as f:
            f.write(label)


    def run(self):
        while True:
            frame = cv2.imread(str(self.frames[self.idx]))
            frame = self._draw_overlays(frame)

            event = self.events.get(self.idx, [None])[0]
            crop = self._load_region_crop(event["region"]) if event else None
            panel = self._render_side_panel(event) if event else None

            if crop is not None:
                frame = cv2.hconcat([frame, crop, panel])

            cv2.putText(
                frame,
                f"Frame {self.idx}/{len(self.frames)-1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Replay Viewer", frame)

            key = cv2.waitKey(30 if self.playing else 0) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord("d"):
                self.idx = min(self.idx + 1, len(self.frames) - 1)
            elif key == ord("a"):
                self.idx = max(self.idx - 1, 0)
            elif key == ord("f"):
                self._jump(direction=1)
            elif key == ord("b"):
                self._jump(direction=-1)
            elif key == ord("y"):
                self._label_event(event, "tp")
            elif key == ord("n"):
                self._label_event(event, "fp")
            elif key == ord("u"):
                self._label_event(event, "uncertain")
            elif key == ord("i"):
                self._label_event(event, "ignore")
            elif key == ord("e"):
                self.export_training_sample(event)
            elif key in CLASS_MAP:
                class_id = CLASS_MAP[key]
                self.export_training_sample(event, class_id)
            elif key == ord(" "):
                self.playing = not self.playing

        cv2.destroyAllWindows()
