from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any


# -----------------------------
# Lint result structure
# -----------------------------

@dataclass
class LintMessage:
    level: str        # "error" | "warning"
    region: str       # region name
    message: str

    def __str__(self):
        return f"[{self.level.upper()}] {self.region}: {self.message}"


# -----------------------------
# Public API
# -----------------------------

def lint_regions(
    regions: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    base_dir: Path | None = None,
) -> List[LintMessage]:
    """
    Lint region definitions against schema and image bounds.
    """
    messages: List[LintMessage] = []
    seen_names = set()

    for r in regions:
        name = r.get("name", "<unnamed>")
        rtype = r.get("type")

        # ---- name ----
        if not r.get("name"):
            messages.append(err(name, "Region missing 'name'"))
        elif name in seen_names:
            messages.append(err(name, "Duplicate region name"))
        seen_names.add(name)

        # ---- type ----
        if rtype not in {"button", "template", "ocr", "hybrid"}:
            messages.append(err(name, f"Unknown region type '{rtype}'"))
            continue

        # ---- rect ----
        rect = r.get("rect")
        if not valid_rect(rect):
            messages.append(err(name, "Invalid rect; expected [x, y, w, h]"))
        else:
            messages.extend(lint_rect_bounds(name, rect, img_w, img_h))

        # ---- per-type checks ----
        if rtype == "template":
            messages.extend(lint_template(name, r, base_dir))

        elif rtype == "ocr":
            messages.extend(lint_ocr(name, r))

        elif rtype == "hybrid":
            messages.extend(lint_hybrid(name, r, base_dir))

    return messages


# -----------------------------
# Rect validation
# -----------------------------

def valid_rect(rect) -> bool:
    return (
        isinstance(rect, list)
        and len(rect) == 4
        and all(isinstance(v, (int, float)) for v in rect)
        and rect[2] > 0
        and rect[3] > 0
    )


def lint_rect_bounds(name: str, rect, img_w: int, img_h: int) -> List[LintMessage]:
    x, y, w, h = rect
    msgs = []

    if x < 0 or y < 0:
        msgs.append(warn(name, "Rect has negative origin"))

    if x + w > img_w or y + h > img_h:
        msgs.append(warn(name, "Rect extends outside image bounds"))

    return msgs


# -----------------------------
# Template linting
# -----------------------------

def lint_template(name: str, r: Dict, base_dir: Path | None) -> List[LintMessage]:
    msgs = []
    tmpl = r.get("template")

    if not tmpl:
        return [err(name, "Template region missing 'template' block")]

    img = tmpl.get("image")
    if not img:
        msgs.append(err(name, "Template missing 'image'"))
    elif base_dir:
        p = (base_dir / img).resolve()
        if not p.exists():
            msgs.append(err(name, f"Template image not found: {img}"))

    thresh = tmpl.get("threshold", 0.8)
    if not (0.0 < thresh <= 1.0):
        msgs.append(warn(name, f"Template threshold out of range: {thresh}"))

    return msgs


# -----------------------------
# OCR linting
# -----------------------------

def lint_ocr(name: str, r: Dict) -> List[LintMessage]:
    msgs = []
    ocr = r.get("ocr")

    if not ocr:
        return [err(name, "OCR region missing 'ocr' block")]

    text = ocr.get("text")
    if not text:
        msgs.append(err(name, "OCR missing 'text'"))

    match = ocr.get("match", "contains")
    if match not in {"contains", "exact", "regex"}:
        msgs.append(warn(name, f"Unknown OCR match mode '{match}'"))

    conf = ocr.get("confidence", 0.5)
    if not (0.0 <= conf <= 1.0):
        msgs.append(warn(name, f"OCR confidence out of range: {conf}"))

    return msgs


# -----------------------------
# Hybrid linting (⭐ IMPORTANT)
# -----------------------------

def lint_hybrid(name: str, r: Dict, base_dir: Path | None) -> List[LintMessage]:
    msgs = []

    has_template = "template" in r
    has_ocr = "ocr" in r

    if not has_template:
        msgs.append(err(name, "Hybrid region missing 'template'"))
    else:
        msgs.extend(lint_template(name, r, base_dir))

    if not has_ocr:
        msgs.append(err(name, "Hybrid region missing 'ocr'"))
    else:
        msgs.extend(lint_ocr(name, r))

    logic = r.get("logic")
    if not logic:
        msgs.append(warn(name, "Hybrid region missing 'logic' block"))
        return msgs

    require = logic.get("require")
    if not isinstance(require, list):
        msgs.append(err(name, "Hybrid logic.require must be a list"))
        return msgs

    valid_keys = {"template", "ocr"}
    for k in require:
        if k not in valid_keys:
            msgs.append(err(name, f"Invalid hybrid requirement '{k}'"))

    if not require:
        msgs.append(warn(name, "Hybrid logic.require is empty"))

    if len(require) == 1:
        msgs.append(warn(
            name,
            f"Hybrid requires only '{require[0]}'; consider simplifying type"
        ))

    agg = logic.get("aggregate", "min")
    if agg not in {"min", "mean", "product"}:
        msgs.append(warn(name, f"Unknown hybrid aggregate '{agg}'"))

    return msgs


# -----------------------------
# Helpers
# -----------------------------

def err(region: str, msg: str) -> LintMessage:
    return LintMessage("error", region, msg)


def warn(region: str, msg: str) -> LintMessage:
    return LintMessage("warning", region, msg)
