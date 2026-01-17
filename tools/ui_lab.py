import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QListWidget,
    QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QDockWidget, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt6.QtCore import Qt

import easyocr


# =========================
# Data model
# =========================

@dataclass
class Region:
    name: str
    rect: list  # [x,y,w,h]
    type: str = "template"  # template | ocr | hybrid
    template_image: str | None = None
    annotation: str = ""
    click: dict | None = None

    # runtime
    template_conf: float = 0.0
    ocr_text: str = ""
    matched: bool = False
    template_heatmap: np.ndarray | None = None


# =========================
# Utility functions
# =========================

def load_regions_yaml(path: Path):
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f) or []
    regions = []
    for r in data:
        regions.append(Region(**r))
    return regions


def save_regions_yaml(path: Path, regions):
    out = []
    for r in regions:
        d = {
            "name": r.name,
            "rect": r.rect,
            "type": r.type,
            "annotation": r.annotation
        }
        if r.template_image:
            d["template_image"] = r.template_image
        if r.click:
            d["click"] = r.click
        out.append(d)
    with open(path, "w") as f:
        yaml.safe_dump(out, f)


def match_template_gray(frame, region: Region):
    region.template_conf = 0.0
    region.template_heatmap = None

    if not region.template_image:
        return

    tpl = cv2.imread(region.template_image, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        return

    x, y, w, h = region.rect
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return

    roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ROI must be >= template
    if roi_g.shape[0] < tpl.shape[0] or roi_g.shape[1] < tpl.shape[1]:
        return

    res = cv2.matchTemplate(roi_g, tpl, cv2.TM_CCOEFF_NORMED)
    region.template_conf = float(res.max())
    region.template_heatmap = res


def run_ocr(frame, region: Region, reader):
    x, y, w, h = region.rect
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        region.ocr_text = ""
        return

    results = reader.readtext(roi, detail=0)
    region.ocr_text = " ".join(results)


def analyze_region(frame, region: Region, reader):
    region.matched = False

    if region.type in ("template", "hybrid"):
        match_template_gray(frame, region)

    if region.type in ("ocr", "hybrid"):
        run_ocr(frame, region, reader)

    if region.type == "template":
        region.matched = region.template_conf >= 0.8
    elif region.type == "ocr":
        region.matched = len(region.ocr_text) > 0
    elif region.type == "hybrid":
        region.matched = (
            region.template_conf >= 0.8 and len(region.ocr_text) > 0
        )


# =========================
# Canvas widget
# =========================

class ImageCanvas(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.main = parent
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.main.current_img is None:
            return

        painter = QPainter(self)

        # ---- HEATMAP ----
        if (
            self.main.show_heatmap
            and self.main.selected_region
            and self.main.selected_region.template_heatmap is not None
        ):
            r = self.main.selected_region
            heat = r.template_heatmap

            # Normalize to 0–255
            heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
            heat_norm = heat_norm.astype(np.uint8)

            # Apply colormap
            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

            # Resize to ROI size
            x, y, w, h = r.rect
            heat_color = cv2.resize(heat_color, (w, h))

            # Convert to QImage
            qimg = QImage(
                heat_color.data,
                w, h,
                3 * w,
                QImage.Format.Format_BGR888
            )

            painter.setOpacity(0.6)
            painter.drawImage(x, y, qimg)
            painter.setOpacity(1.0)

    # ---- REGIONS ----
    for r in self.main.regions:
        x, y, w, h = r.rect
        color = QColor(0, 255, 0) if r.matched else QColor(255, 0, 0)
        painter.setPen(QPen(color, 2))
        painter.drawRect(x, y, w, h)

        label = f"{r.name} T:{r.template_conf:.2f}"
        if r.ocr_text:
            label += f' OCR:"{r.ocr_text}"'
        painter.drawText(x + 4, y + 14, label)



# =========================
# Main window
# =========================

class UILabMainWindow(QMainWindow):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.setWindowTitle("UI Vision Lab")

        self.run_dir = run_dir
        self.frames_dir = run_dir / "frames"
        self.regions_path = run_dir / "regions.yaml"

        self.reader = easyocr.Reader(["en"], gpu=False)

        self.frames = sorted(self.frames_dir.glob("*.png"))
        self.frame_idx = 0
        self.current_img = None

        self.regions = load_regions_yaml(self.regions_path)

        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)
        self.show_heatmap = False
        self.selected_region = None

        self._build_ui()
        self._load_frame(0)

    # ---------- UI ----------

    def _build_ui(self):
        # toolbar
        bar = QWidget()
        layout = QHBoxLayout(bar)

        btn_prev = QPushButton("◀ Prev")
        btn_next = QPushButton("Next ▶")
        btn_save = QPushButton("Save Regions")

        btn_prev.clicked.connect(self.prev_frame)
        btn_next.clicked.connect(self.next_frame)
        btn_save.clicked.connect(self.save_regions)

        btn_heatmap = QPushButton("Heatmap")
        btn_heatmap.setCheckable(True)
        btn_heatmap.toggled.connect(self.toggle_heatmap)
        
        layout.addWidget(btn_heatmap)
        layout.addWidget(btn_prev)
        layout.addWidget(btn_next)
        layout.addWidget(btn_save)        

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._wrap(bar))

        # frame list
        dock = QDockWidget("Frames", self)
        self.frame_list = QListWidget()
        for f in self.frames:
            self.frame_list.addItem(f.name)
        self.frame_list.currentRowChanged.connect(self._load_frame)
        dock.setWidget(self.frame_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _wrap(self, widget):
        from PyQt6.QtWidgets import QToolBar
        tb = QToolBar()
        tb.addWidget(widget)
        return tb

    # ---------- Frame logic ----------

    def _load_frame(self, idx):
        if not self.frames:
            return
        idx = max(0, min(idx, len(self.frames) - 1))
        self.frame_idx = idx
        path = self.frames[idx]

        self.current_img = cv2.imread(str(path))
        self._reanalyze()

        h, w, _ = self.current_img.shape
        qimg = QImage(
            self.current_img.data, w, h, 3 * w, QImage.Format.Format_BGR888
        )
        self.canvas.setPixmap(QPixmap.fromImage(qimg))
        self.canvas.resize(w, h)

        self.frame_list.blockSignals(True)
        self.frame_list.setCurrentRow(idx)
        self.frame_list.blockSignals(False)

    def _reanalyze(self):
        if self.regions:
            self.selected_region = self.regions[0]
        for r in self.regions:
            analyze_region(self.current_img, r, self.reader)
        self.canvas.update()

    def prev_frame(self):
        self._load_frame(self.frame_idx - 1)

    def next_frame(self):
        self._load_frame(self.frame_idx + 1)

    def toggle_heatmap(self, checked):
        self.show_heatmap = checked
        self.canvas.update()

    # ---------- Regions ----------

    def save_regions(self):
        save_regions_yaml(self.regions_path, self.regions)
        QMessageBox.information(self, "Saved", "regions.yaml updated")


# =========================
# Entry point
# =========================

def find_default_run_dir():
    base = Path("debug_runs/run_latest")
    return base if base.exists() else None


if __name__ == "__main__":
    app = QApplication(sys.argv)

    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else find_default_run_dir()
    if not run_dir or not run_dir.exists():
        QMessageBox.critical(None, "Error", "No valid run_dir found")
        sys.exit(1)

    win = UILabMainWindow(run_dir)
    win.show()
    sys.exit(app.exec())
