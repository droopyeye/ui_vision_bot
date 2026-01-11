from debug.recorder import FrameRecorder

if __name__ == "__main__":
    print("Starting capture… Ctrl+C to stop")
    rec = FrameRecorder(fps=5)
    rec.run()
    print(f"Capture complete. Frames saved to: {rec.run_dir}")