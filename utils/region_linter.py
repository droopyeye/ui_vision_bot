import os

ALLOWED_MODES = {"template", "ocr", "hybrid", "yolo"}

def lint_regions(config, template_dir="config/templates"):
    errors = []

    regions = config.get("regions", {})
    if not regions:
        errors.append("No regions defined")

    for name, r in regions.items():
        mode = r.get("mode")

        if mode not in ALLOWED_MODES:
            errors.append(f"[{name}] Invalid or missing mode: {mode}")
            continue

        # YOLO mode
        if mode == "yolo":
            if "class" not in r:
                errors.append(f"[{name}] YOLO mode requires 'class'")
            forbidden = {"rect", "template", "ocr"}
            for f in forbidden:
                if f in r:
                    errors.append(f"[{name}] '{f}' not allowed in YOLO mode")
            continue

        # Non-YOLO modes require rect
        rect = r.get("rect")
        if not rect:
            errors.append(f"[{name}] Missing rect for mode '{mode}'")
        else:
            for k in ("x", "y", "w", "h"):
                if k not in rect:
                    errors.append(f"[{name}] rect missing '{k}'")

        # Template checks
        if mode in ("template", "hybrid"):
            t = r.get("template")
            if not t or "file" not in t:
                errors.append(f"[{name}] Template mode requires template.file")
            else:
                path = os.path.join(template_dir, t["file"])
                if not os.path.exists(path):
                    errors.append(f"[{name}] Missing template file: {path}")

            if "threshold" not in t:
                errors.append(f"[{name}] Template missing threshold")

        # OCR checks
        if mode in ("ocr", "hybrid"):
            o = r.get("ocr")
            if not o or "expected" not in o:
                errors.append(f"[{name}] OCR mode requires ocr.expected")
            if o and not isinstance(o.get("expected"), list):
                errors.append(f"[{name}] ocr.expected must be list")

        # Forbidden combinations
        if mode == "template" and "ocr" in r:
            errors.append(f"[{name}] OCR not allowed in template-only mode")

    return errors
