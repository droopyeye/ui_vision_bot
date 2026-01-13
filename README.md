# ui_vision_bot
Python UI lab for image training automation bot

# main.py
- Loads regions.yaml
- Runs template, OCR, and hybrid detection
- Uses hybrid confidence aggregation
- Is safe to import from live_runner.py
- Does not click (main.py = vision logic only)
- Plays nicely with UI Lab and replay tools

UI Lab ─┐
        ├── analyze_frame() ──> overlays / previews
Replay ─┤
        ├── live_runner.py ──> clicks
Policy ─┘