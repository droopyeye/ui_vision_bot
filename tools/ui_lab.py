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
        self.template_confidence = 0.0
        self.ocr_confidence = 0.0
        self.hybrid_confidence = 0.0
        self.matched = False  # True if template/OCR passes threshold

        # template match location (offset within ROI)
        self.template_match_loc = None  # (x_offset, y_offset) within the region rect
        self.template_size = None  # (width, height) of matched template

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

        # Left: frame list browser
        left = QVBoxLayout()
        layout.addLayout(left, 1)

        left.addWidget(QLabel("Frames"))
        self.frame_list = QListWidget()
        self.frame_list.currentItemChanged.connect(self._select_frame)
        left.addWidget(self.frame_list)

        # Populate frame list
        for i, frame_path in enumerate(self.frames):
            item = QListWidgetItem(frame_path.name)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.frame_list.addItem(item)

        # Center: image
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

        # REGION rect edits
        right.addWidget(QLabel("Rect (x, y, w, h)"))
        self.rect_x = QSpinBox(); self.rect_x.setMaximum(10000)
        self.rect_y = QSpinBox(); self.rect_y.setMaximum(10000)
        self.rect_w = QSpinBox(); self.rect_w.setMaximum(5000)
        self.rect_h = QSpinBox(); self.rect_h.setMaximum(5000)
        self.rect_x.valueChanged.connect(self._update_rect_from_ui)
        self.rect_y.valueChanged.connect(self._update_rect_from_ui)
        self.rect_w.valueChanged.connect(self._update_rect_from_ui)
        self.rect_h.valueChanged.connect(self._update_rect_from_ui)
        right.addWidget(self.rect_x)
        right.addWidget(self.rect_y)
        right.addWidget(self.rect_w)
        right.addWidget(self.rect_h)

        btns = QHBoxLayout()
        add_btn = QPushButton("Add Region")
        add_btn.clicked.connect(self._add_region)
        draw_btn = QPushButton("Draw Rect")
        draw_btn.clicked.connect(self._toggle_draw)
        save_btn = QPushButton("Save regions.yaml")
        save_btn.clicked.connect(self._save_regions)
        btn_ocr = QPushButton("Analyze Frame")
        btn_ocr.clicked.connect(self.update_region_analysis)
        right.addWidget(btn_ocr)
        btn_template_overlay = QPushButton("Run Template Overlay")
        btn_template_overlay.clicked.connect(self.run_template_overlay)
        right.addWidget(btn_template_overlay)

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

        # Set first frame as selected (after all UI elements are created)
        if self.frames:
            self.frame_list.setCurrentRow(0)

    # ---------------- Frame ----------------

    def _select_frame(self, item):
        """Handle frame selection from the list"""
        if not item:
            return
        self.idx = item.data(Qt.ItemDataRole.UserRole)
        self._load_frame()

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

        # Set rect spin boxes
        x, y, w, h = r.rect
        self.rect_x.setValue(x)
        self.rect_y.setValue(y)
        self.rect_w.setValue(w)
        self.rect_h.setValue(h)

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

    def _update_rect_from_ui(self):
        item = self.region_list.currentItem()
        if not item:
            return
        r = self.regions[self.region_list.row(item)]
        r.rect = [
            self.rect_x.value(),
            self.rect_y.value(),
            self.rect_w.value(),
            self.rect_h.value()
        ]
        self._draw_regions()

    def _draw_confidence_overlay(self, frame):
        for r in self.regions:
            x, y, w, h = r.rect
            conf_text = ""
            if r.type in ["ocr", "hybrid"]:
                conf_text += f"OCR: {r.ocr_confidence:.2f} "
            if r.type in ["template", "hybrid"]:
                conf_text += f"Tmpl: {r.template_confidence:.2f} "
            if r.type=="hybrid":
                conf_text += f"Final: {r.hybrid_confidence:.2f}"

            label = self.view.scene().addText(conf_text)
            label.setDefaultTextColor(QColor("yellow"))
            label.setPos(x, y-20)

    def run_template_overlay(self):
        """
        For each region with a template, perform template matching and
        draw the match result as a rectangle over the frame.
        """
        if not hasattr(self, "current_img"):
            return

        frame = self.current_img.copy()
        self.view.scene().clear()
        pix = QPixmap.fromImage(cv_to_qimage(frame))
        self.view.scene().addPixmap(pix)

        for r in self.regions:
            if not r.template_image:
                continue

            # Load template
            tmpl_path = (self.run_dir / r.template_image).resolve()
            tmpl = cv2.imread(str(tmpl_path), cv2.IMREAD_UNCHANGED)
            if tmpl is None:
                print(f"⚠️ Template not found for region {r.name}")
                continue

            # ROI from region rect
            x, y, w, h = r.rect
            roi = frame[y:y+h, x:x+w]

            # Convert to grayscale for robustness
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

            # Run template match
            res = cv2.matchTemplate(roi_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            r.template_confidence = float(max_val)

            # Store match location and template size
            r.template_match_loc = max_loc
            r.template_size = (tmpl.shape[1], tmpl.shape[0])

            # Draw template match rectangle
            top_left = (x + max_loc[0], y + max_loc[1])
            bottom_right = (top_left[0] + tmpl.shape[1], top_left[1] + tmpl.shape[0])
            rect_item = self.view.scene().addRect(
                top_left[0], top_left[1],
                tmpl.shape[1], tmpl.shape[0],
                QPen(QColor("red"), 2)
            )

            # Draw click point if enabled
            if self.preview_clicks and r.click:
                offset = r.click.get("offset", [0, 0])
                # Use template match location
                match_x, match_y = r.template_match_loc
                tmpl_w, tmpl_h = r.template_size
                # Calculate center of matched template
                cx = x + match_x + tmpl_w // 2
                cy = y + match_y + tmpl_h // 2
                cx += offset[0]
                cy += offset[1]
                self.view.scene().addEllipse(cx-3, cy-3, 6, 6,
                                            QPen(QColor("blue")),
                                            QColor(0,0,255,100))

            # Add confidence label
            label_text = f"{r.name} conf: {r.template_confidence:.2f}"
            label_item = self.view.scene().addText(label_text)
            label_item.setDefaultTextColor(QColor("yellow"))
            label_item.setPos(x, y-20)

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
        self.view.scene().clear()
        pix = QPixmap.fromImage(cv_to_qimage(self.current_img))
        self.view.scene().addPixmap(pix)

        for r in self.regions:
            x, y, w, h = r.rect
            color = QColor("green") if r.matched else QColor("cyan")
            rect_item = QGraphicsRectItem(x, y, w, h)
            rect_item.setPen(QPen(color, 2))
            self.view.scene().addItem(rect_item)

            # Click point
            if self.preview_clicks and r.click:
                offset = r.click.get("offset", [0,0])

                # Use template match location for template/hybrid types
                if r.type in ["template", "hybrid"] and r.template_match_loc and r.template_size:
                    match_x, match_y = r.template_match_loc
                    tmpl_w, tmpl_h = r.template_size
                    # Calculate center of matched template
                    cx = x + match_x + tmpl_w // 2
                    cy = y + match_y + tmpl_h // 2
                else:
                    # Fallback to region center for OCR-only or when no match
                    mode = r.click.get("mode", "center")
                    cx, cy = (x + w//2, y + h//2) if mode=="center" else (x, y)

                cx += offset[0]; cy += offset[1]
                self.view.scene().addEllipse(cx-3, cy-3, 6, 6,
                                            QPen(QColor("red")),
                                            QColor(255,0,0,100))

            # Confidence overlay
            conf_text = f"OCR: {r.ocr_confidence:.2f} | Tmpl: {r.template_confidence:.2f}"
            if r.type == "hybrid":
                conf_text += f" | Final: {r.hybrid_confidence:.2f}"
            label = self.view.scene().addText(conf_text)
            label.setDefaultTextColor(QColor("yellow"))
            label.setPos(x, y-20)

    def update_region_analysis(self):
        frame = self.current_img.copy()

        for r in self.regions:
            # ---- OCR confidence ----
            if r.type in ["ocr", "hybrid"]:
                result = self.reader.readtext(frame[r.rect[1]:r.rect[1]+r.rect[3],
                                                r.rect[0]:r.rect[0]+r.rect[2]])
                r.ocr_confidence = max([conf for _, text, conf in result], default=0.0)

            # ---- Template confidence ----
            if r.type in ["template", "hybrid"] and r.template_image:
                tmpl_path = (self.run_dir / r.template_image).resolve()
                tmpl = cv2.imread(str(tmpl_path), cv2.IMREAD_UNCHANGED)
                if tmpl is None:
                    r.template_confidence = 0.0
                else:
                    # Make sure ROI is large enough
                    x, y, w, h = r.rect
                    roi = self.current_img[y:y+h, x:x+w]
                    if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
                        print(f"ROI smaller than template for {r.name}")
                        r.template_confidence = 0.0
                    else:
                        # Convert both to grayscale for consistent matching
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                        res = cv2.matchTemplate(roi_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        r.template_confidence = float(max_val)
                        # Store match location and template size
                        r.template_match_loc = max_loc
                        r.template_size = (tmpl.shape[1], tmpl.shape[0])
                        print(f"{r.name} ROI size: {roi.shape}, Template size: {tmpl.shape}, Confidence: {r.template_confidence}")

            # ---- Hybrid aggregation ----
            if r.type == "hybrid":
                r.hybrid_confidence = (r.ocr_confidence + r.template_confidence) / 2.0

            # Matched threshold (example: 0.7)
            threshold = 0.7
            if r.type == "hybrid":
                r.matched = r.hybrid_confidence >= threshold
            elif r.type == "ocr":
                r.matched = r.ocr_confidence >= threshold
            elif r.type == "template":
                r.matched = r.template_confidence >= threshold

        self._draw_regions()

    # ---------------- Entry ----------------

def main():
    # Default to debug_runs/run_latest if no run_dir specified
    if len(sys.argv) < 2:
        run_dir = Path(__file__).parent.parent / "debug_runs" / "run_latest"
        if not run_dir.exists():
            print("Usage: ui_lab.py <run_dir>")
            print(f"Default run_dir not found: {run_dir}")
            sys.exit(1)
        print(f"Using default run_dir: {run_dir}")
    else:
        run_dir = sys.argv[1]

    app = QApplication(sys.argv)
    win = UILabMainWindow(run_dir)
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
