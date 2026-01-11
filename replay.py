# replay.py
import yaml
from debug.replay_viewer import ReplayViewer

regions = yaml.safe_load(open("config/regions.yaml"))
viewer = ReplayViewer(
    run_dir="debug_runs/run_2026-01-10_08-42-15",
    regions_config=regions,
)
viewer.run()
