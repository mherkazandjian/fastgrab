"""Live capture-progress window shown while the recorder runs.

Used by the ``--gui`` flow, where ``fastgrab-record`` is typically
triggered from a hotkey and has no controlling terminal to print to. The
recorder loop runs in a worker thread; this window polls a shared counter
via ``root.after`` and shows the frame count, elapsed time, and achieved
fps, plus a Stop button. Tk is imported lazily so the module stays
importable in headless contexts.

The window is deliberately *not* forced always-on-top: with a non-zero
countdown it opens, counts down, and lets you minimise it or drag it out
of shot before the first frame is captured.
"""
import math
import threading


def _place_near(root, bbox, win_w, win_h):
    """Position the window just outside the captured region if possible.

    The window is on-screen and top-most, so anything it overlaps ends up
    baked into the recording. We try below the region, then above, then
    fall back to the top-left corner (unavoidable for a fullscreen grab —
    there is no off-region space to hide in).
    """
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    margin = 12
    if bbox is not None:
        x, y, _w, h = bbox
        if y + h + margin + win_h <= screen_h:
            px, py = x, y + h + margin
        elif y - margin - win_h >= 0:
            px, py = x, y - margin - win_h
        else:
            px, py = margin, margin
    else:
        px, py = margin, margin
    px = max(0, min(px, screen_w - win_w))
    py = max(0, min(py, screen_h - win_h))
    root.geometry("+{}+{}".format(int(px), int(py)))


def record_with_progress(recorder, bbox=None, duration=None, stop_event=None,
                         countdown=0):
    """Run ``recorder.record()`` in a thread behind a live-progress window.

    Returns the recorder's stats dict, or re-raises whatever the recorder
    raised (e.g. ffmpeg missing) after surfacing it in a dialog. The passed
    ``stop_event`` is shared with the CLI's signal handlers; the window's
    Stop button and its close box set it too.

    ``countdown`` (seconds) delays the first frame; the window shows the
    remaining seconds and, being non-topmost, can be minimised or moved off
    screen during that window so it doesn't end up in the recording.
    """
    import tkinter as tk

    if stop_event is None:
        stop_event = threading.Event()

    # Written by the worker thread, read by the Tk poller. Only plain data
    # crosses the boundary — Tk itself is touched from the main thread only.
    shared = {"frames": 0, "elapsed": 0.0, "countdown": None,
              "stats": None, "error": None}

    def _on_progress(frames, elapsed):
        shared["frames"] = frames
        shared["elapsed"] = elapsed

    def _on_countdown(remaining):
        shared["countdown"] = remaining

    def _worker():
        try:
            shared["stats"] = recorder.record(
                duration=duration, stop_event=stop_event,
                on_progress=_on_progress,
                countdown=countdown, on_countdown=_on_countdown,
            )
        except Exception as exc:  # surfaced by the poller / caller below
            shared["error"] = exc

    root = tk.Tk()
    root.title("fastgrab — recording")
    # Not topmost on purpose: the point of the countdown is to let you send
    # this window behind everything else before capture starts.
    root.resizable(False, False)

    status_var = tk.StringVar(value="Starting…")
    frames_var = tk.StringVar(value="0 frames")
    detail_var = tk.StringVar(value="")

    tk.Label(root, textvariable=status_var, font=("TkDefaultFont", 12)).grid(
        row=0, column=0, padx=24, pady=(14, 2)
    )
    tk.Label(root, textvariable=frames_var, font=("TkFixedFont", 22)).grid(
        row=1, column=0, padx=24
    )
    tk.Label(root, textvariable=detail_var, fg="#666").grid(
        row=2, column=0, padx=24, pady=(0, 4)
    )

    def _stop():
        stop_event.set()

    stop_btn = tk.Button(root, text="Stop", width=12, command=_stop)
    stop_btn.grid(row=3, column=0, pady=(4, 14))
    root.bind("<Escape>", lambda _e: _stop())
    # The window-manager close box should stop cleanly, not kill the encode.
    root.protocol("WM_DELETE_WINDOW", _stop)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    def _tick():
        if not worker.is_alive():
            if shared["error"] is None:
                # Brief confirmation so a quick clip doesn't just vanish.
                status_var.set("Saved" if shared["frames"] else "Cancelled")
                frames_var.set("{} frames".format(shared["frames"]))
                stop_btn.config(state="disabled")
                root.after(1200, root.destroy)
            else:
                root.destroy()
            return

        cd = shared["countdown"]
        if cd is not None and cd > 0.05 and shared["frames"] == 0:
            status_var.set("Starting in {}s…".format(int(math.ceil(cd))))
            frames_var.set("get ready")
            detail_var.set("minimise or move this window out of shot")
        else:
            frames = shared["frames"]
            elapsed = shared["elapsed"]
            fps = (frames / elapsed) if elapsed > 0 else 0.0
            frames_var.set("{} frames".format(frames))
            detail_var.set("{:.1f}s · {:.1f} fps".format(elapsed, fps))
            # Between Stop and ffmpeg finishing its flush the worker is
            # still alive; say so rather than freezing on "Recording…".
            if stop_event.is_set():
                status_var.set("Finalizing…")
                stop_btn.config(state="disabled")
            else:
                status_var.set("Recording…")
        root.after(100, _tick)

    # Size the window from its content, then park it clear of the capture.
    root.update_idletasks()
    _place_near(root, bbox, root.winfo_reqwidth(), root.winfo_reqheight())

    root.after(100, _tick)
    root.mainloop()

    worker.join(timeout=5.0)
    if shared["error"] is not None:
        _show_error(shared["error"])
        raise shared["error"]
    return shared["stats"]


def _show_error(exc):
    """Pop a modal error dialog — the hotkey flow has no terminal to read."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        err_root = tk.Tk()
        err_root.withdraw()
        messagebox.showerror("fastgrab — recording failed", str(exc))
        err_root.destroy()
    except Exception:
        pass
