import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QFileDialog, QHBoxLayout, QListWidget,
    QMessageBox, QDockWidget, QToolBar
)
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PyQt6.QtCore import Qt, QRect, QPoint, QTimer

import easyocr


# =========================
# Region model
# =========================

@dataclass
class Region:
    name: str
    type: str
    x: int
    y: int
    w: int
    h: int


# =========================
# YAML IO
# =========================

def load_regions_yaml(path: Path) -> List[Region]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or []

    return [
        Region(
            name=r["name"],
            type=r["type"],
            x=int(r["x"]),
            y=int(r["y"]),
            w=int(r["w"]),
            h=int(r["h"]),
        )
        for r in data
    ]


def save_regions_yaml(path: Path, regions: List[Region]):
    with open(path, "w") as f:
        yaml.safe_dump(
            [
                dict(
                    name=r.name,
                    type=r.type,
                    x=r.x,
                    y=r.y,
                    w=r.w,
                    h=r.h,
                )
                for r in regions
            ],
            f,
        )


# =========================
# Region linter (with flags)
# =========================

def lint_regions(
    regions: List[Region], img_w: int, img_h: int
) -> Dict[str, List[str]]:
    issues: Dict[str, List[str]] = {}

    def add(region: Region, msg: str):
        issues.setdefault(region.name, []).append(msg)

    for r in regions:
        if r.w <= 0 or r.h <= 0:
            add(r, "zero or negative size")

        if r.x < 0 or r.y < 0 or r.x + r.w > img_w or r.y + r.h > img_h:
            add(r, "out of bounds")

    for i in range(len(regions)):
        a = regions[i]
        ra = QRect(a.x, a.y, a.w, a.h)
        for j in range(i + 1, len(regions)):
            b = regions[j]
            rb = QRect(b.x, b.y, b.w, b.h)
            if ra.intersects(rb):
                add(a, f"overlaps {b.name}")
                add(b, f"overlaps {a.name}")

    return issues


# =========================
# Canvas
# =========================

class FrameCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.regions: List[Region] = []
        self.lint_issues: Dict[str, List[str]] = {}

        self.drawing = False
        self.start = QPoint()
        self.end = QPoint()

    def set_frame(self, frame: np.ndarray):
        self.frame = frame
        self.update()

    def paintEvent(self, event):
        if self.frame is None:
            return

        h, w, _ = self.frame.shape
        img = QImage(self.frame.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(img)

        painter = QPainter(self)
        painter.drawPixmap(0, 0, pix)

        for r in self.regions:
            issues = self.lint_issues.get(r.name, [])

            if any("overlaps" in i for i in issues):
                pen = QPen(QColor(255, 165, 0), 3)  # orange
            elif issues:
                pen = QPen(QColor(255, 0, 0), 3)  # red
            else:
                pen = QPen(QColor(0, 255, 0), 2)  # green

            painter.setPen(pen)
            painter.drawRect(r.x, r.y, r.w, r.h)

            label = r.name
            if issues:
                label += " ⚠"

            painter.drawText(r.x + 3, r.y + 14, label)

        if self.drawing:
            pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self.start, self.end).normalized())

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.start = e.position().toPoint()
            self.end = self.start

    def mouseMoveEvent(self, e):
        if self.drawing:
            self.end = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if not self.drawing:
            return

        self.drawing = False
        rect = QRect(self.start, self.end).normalized()

        if rect.width() > 5 and rect.height() > 5:
            self.regions.append(
                Region(
                    name=f"region_{len(self.regions)+1}",
                    type="button",
                    x=rect.x(),
                    y=rect.y(),
                    w=rect.width(),
                    h=rect.height(),
                )
            )
        self.update()


# =========================
# Main window
# =========================

class UILabMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UI Vision Lab")

        self.run_dir: Path | None = None
        self.frames = []
        self.idx = 0
        self.current_frame = None

        self.reader = easyocr.Reader(["en"], gpu=False)

        self.canvas = FrameCanvas()
        self.setCentralWidget(self.canvas)

        # ✅ CREATE TIMER FIRST
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # THEN BUILD UI
        self._build_ui()

    def _build_ui(self):
        bar_widget = QWidget()
        layout = QHBoxLayout(bar_widget)

        for label, fn in [
            ("Open Run", self.open_run),
            ("Play", lambda: self.timer.start(200)),
            ("Stop", self.timer.stop),
            ("Save Regions", self.save_regions),
            ("Lint Regions", self.run_linter),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(fn)
            layout.addWidget(btn)

        # Create and store the toolbar
        self.toolbar = QToolBar("Main")
        self.toolbar.addWidget(bar_widget)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Linter dock (unchanged)
        self.lint_list = QListWidget()
        dock = QDockWidget("Region Linter", self)
        dock.setWidget(self.lint_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def open_run(self):
        path = QFileDialog.getExistingDirectory(self, "Open Run")
        if not path:
            return

        self.run_dir = Path(path)
        self.frames = sorted((self.run_dir / "frames").glob("*.png"))

        if not self.frames:
            QMessageBox.warning(self, "Error", "No frames found")
            return

        rp = self.run_dir / "regions.yaml"
        self.canvas.regions = load_regions_yaml(rp) if rp.exists() else []

        self.idx = 0
        self.load_frame()

    def load_frame(self):
        self.current_frame = cv2.imread(str(self.frames[self.idx]))
        self.canvas.set_frame(self.current_frame)

    def next_frame(self):
        self.idx = (self.idx + 1) % len(self.frames)
        self.load_frame()

    def save_regions(self):
        if self.run_dir:
            save_regions_yaml(self.run_dir / "regions.yaml", self.canvas.regions)

    def run_linter(self):
        self.lint_list.clear()

        if self.current_frame is None:
            return

        h, w, _ = self.current_frame.shape
        issues = lint_regions(self.canvas.regions, w, h)
        self.canvas.lint_issues = issues
        self.canvas.update()

        if not issues:
            self.lint_list.addItem("✓ No issues found")
        else:
            for r, msgs in issues.items():
                for m in msgs:
                    self.lint_list.addItem(f"{r}: {m}")


# =========================
# Entry
# =========================

def main():
    app = QApplication(sys.argv)
    win = UILabMainWindow()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
