"""Record the full screen for a few seconds to an mp4.

Run inside the dev container:

    docker compose up -d dev
    docker compose exec dev python examples/record_demo.py
"""
from fastgrab.recording import Recorder


if __name__ == "__main__":
    rec = Recorder(output_path="demo.mp4", fps=30, backend="x11")
    stats = rec.record(duration=3.0)
    print(
        "recorded {frames} frames in {elapsed:.2f}s "
        "({fps:.1f} fps) → {output}".format(
            frames=stats["frames"],
            elapsed=stats["elapsed_seconds"],
            fps=stats["achieved_fps"],
            output=stats["output"],
        )
    )
