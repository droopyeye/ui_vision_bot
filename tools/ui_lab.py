import sys
import yaml
import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QTextEdit,
    QComboBox, QSpinBox, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QCheckBox, QMessageBox
)

import easyocr


# ----------------------------
# Utilities
# ----------------------------

def cv_to_qimage(img):
    h, w, ch = img.shape
    bytes_per_line = ch * w
    return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)


# ----------------------------
# Region Model
# ----------------------------

class Region:
    def __init__(self, data):
        self.name = data.get("name", "")
        self.annotation = data.get("annotation", "")
        self.type = data.get("type", "ocr")
        self.rect = data.get("rect", [0, 0, 100, 100])
        self.template_image = data.get("template_image")
        self.ocr_text = data.get("ocr_text", "")
        self.click = data.get("click", {"mode": "center", "offset": [0, 0]})

    def to_dict(self):
        return {
            "name": self.name,
            "annotation": self.annotation,
            "type": self.type,
            "rect": self.rect,
            "template_image": self.template_image,
            "ocr_text": self.ocr_text,
            "click": self.click,
        }


# ----------------------------
# Graphics View (Zoom + Draw)
# ----------------------------

class ImageView(QGraphicsView):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setScene(QGraphicsScene())
        self._zoom = 0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if self.parent.draw_mode:
            self.parent.start_rect(self.mapToScene(event.position().toPoint()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.parent.draw_mode:
            self.parent.update_rect(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.parent.draw_mode:
            self.parent.finish_rect()
        super().mouseReleaseEvent(event)


# ----------------------------
# Main UI Lab Window
# ----------------------------

class UILabMainWindow(QMainWindow):
    def __init__(self, run_dir):
        super().__init__()
        self.setWindowTitle("UI Vision Lab")

        self.run_dir = Path(run_dir)
        self.frames = sorted(self.run_dir.glob("frames/*.png"))
        if not self.frames:
            QMessageBox.critical(self, "Error", "No frames found")
            sys.exit(1)

        self.idx = 0
        self.regions = []
        self.draw_mode = False
        self.temp_rect_item = None
        self.preview_clicks = True

        # OCR reader (GPU auto-detect)
        self.reader = easyocr.Reader(["en"], gpu=True)

        self._build_ui()
        self._load_regions()
        self._load_frame()

    # ---------------- UI ----------------

    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        # Left: image
        self.view = ImageView(self)
        layout.addWidget(self.view, 3)

        # Right: controls
        right = QVBoxLayout()
        layout.addLayout(right, 1)

        self.region_list = QListWidget()
        self.region_list.currentItemChanged.connect(self._select_region)
        right.addWidget(self.region_list)

        self.name_edit = QLineEdit()
        self.annotation_edit = QTextEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["ocr", "template", "hybrid"])

        self.ocr_text_edit = QLineEdit()
        self.template_path_edit = QLineEdit()

        right.addWidget(QLabel("Name"))
        right.addWidget(self.name_edit)
        right.addWidget(QLabel("Annotation"))
        right.addWidget(self.annotation_edit)
        right.addWidget(QLabel("Type"))
        right.addWidget(self.type_combo)
        right.addWidget(QLabel("OCR Text"))
        right.addWidget(self.ocr_text_edit)
        right.addWidget(QLabel("Template Image"))
        right.addWidget(self.template_path_edit)

        btns = QHBoxLayout()
        add_btn = QPushButton("Add Region")
        add_btn.clicked.connect(self._add_region)
        draw_btn = QPushButton("Draw Rect")
        draw_btn.clicked.connect(self._toggle_draw)
        save_btn = QPushButton("Save regions.yaml")
        save_btn.clicked.connect(self._save_regions)

        btns.addWidget(add_btn)
        btns.addWidget(draw_btn)
        btns.addWidget(save_btn)
        right.addLayout(btns)

        self.preview_checkbox = QCheckBox("Preview Clicks")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.stateChanged.connect(
            lambda s: setattr(self, "preview_clicks", bool(s))
        )
        right.addWidget(self.preview_checkbox)

    # ---------------- Frame ----------------

    def _load_frame(self):
        img = cv2.imread(str(self.frames[self.idx]))
        self.current_img = img
        self.view.scene().clear()
        pix = QPixmap.fromImage(cv_to_qimage(img))
        self.view.scene().addPixmap(pix)
        self._draw_regions()

    # ---------------- Regions ----------------

    def _load_regions(self):
        path = self.run_dir / "regions.yaml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or []
        self.regions = [Region(d) for d in data]
        self._refresh_region_list()

    def _save_regions(self):
        path = self.run_dir / "regions.yaml"
        yaml.safe_dump([r.to_dict() for r in self.regions], path.open("w"))
        QMessageBox.information(self, "Saved", f"Saved {path}")

    def _refresh_region_list(self):
        self.region_list.clear()
        for r in self.regions:
            item = QListWidgetItem(r.name)
            self.region_list.addItem(item)

    def _select_region(self, item):
        if not item:
            return
        r = self.regions[self.region_list.row(item)]
        self.name_edit.setText(r.name)
        self.annotation_edit.setText(r.annotation)
        self.type_combo.setCurrentText(r.type)
        self.ocr_text_edit.setText(r.ocr_text)
        self.template_path_edit.setText(str(r.template_image or ""))

    def _add_region(self):
        r = Region({"name": f"region_{len(self.regions)}"})
        self.regions.append(r)
        self._refresh_region_list()

    # ---------------- Draw ----------------

    def _toggle_draw(self):
        self.draw_mode = not self.draw_mode

    def start_rect(self, pos):
        self.start_pos = pos
        self.temp_rect_item = QGraphicsRectItem()
        self.temp_rect_item.setPen(QPen(QColor("yellow"), 2))
        self.view.scene().addItem(self.temp_rect_item)

    def update_rect(self, pos):
        if not self.temp_rect_item:
            return
        rect = QRectF(self.start_pos, pos).normalized()
        self.temp_rect_item.setRect(rect)

    def finish_rect(self):
        rect = self.temp_rect_item.rect()
        self.view.scene().removeItem(self.temp_rect_item)
        self.temp_rect_item = None
        self.draw_mode = False

        r = Region({
            "name": f"region_{len(self.regions)}",
            "rect": [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
        })
        self.regions.append(r)
        self._refresh_region_list()
        self._draw_regions()

    def _draw_regions(self):
        for r in self.regions:
            x, y, w, h = r.rect
            color = QColor("green") if r.type == "hybrid" else QColor("cyan")
            item = QGraphicsRectItem(x, y, w, h)
            item.setPen(QPen(color, 2))
            self.view.scene().addItem(item)

    # ---------------- Entry ----------------

def main():
    if len(sys.argv) < 2:
        print("Usage: ui_lab.py <run_dir>")
        sys.exit(1)

    app = QApplication(sys.argv)
    win = UILabMainWindow(sys.argv[1])
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
