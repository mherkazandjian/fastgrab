"""Tkinter config dialog shown after the user has picked a region.

Returns a config dict on Record, or ``None`` on Cancel / window-close.
"""
import os


_FORMATS = ("mp4", "webm", "gif")


def _default_output(fmt: str) -> str:
    # Sit next to the user's home so the file is easy to find. The
    # extension stays in lock-step with the format dropdown.
    return os.path.join(os.path.expanduser("~"), "fastgrab.{}".format(fmt))


def show_config_dialog(bbox=None, defaults: dict = None):
    """Show the modal config form. Returns a config dict or ``None``."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as exc:
        raise RuntimeError(
            "fastgrab-record --gui needs tkinter — install your "
            "distro's python3-tk package"
        ) from exc

    defaults = defaults or {}
    root = tk.Tk()
    root.title("fastgrab — record")
    root.attributes("-topmost", True)

    fmt_var = tk.StringVar(value=defaults.get("format", "mp4"))
    output_var = tk.StringVar(
        value=defaults.get("output") or _default_output(fmt_var.get())
    )
    title_var = tk.StringVar(value=defaults.get("title", ""))
    overlay_var = tk.StringVar(value=defaults.get("overlay_text", ""))
    fps_var = tk.IntVar(value=defaults.get("fps", 30))
    duration_var = tk.StringVar(value=defaults.get("duration", ""))
    countdown_var = tk.IntVar(value=int(defaults.get("countdown", 0)))
    show_clicks_var = tk.BooleanVar(value=defaults.get("show_clicks", False))

    pad = {"padx": 8, "pady": 4}
    row = 0

    def label(text):
        nonlocal row
        tk.Label(root, text=text, anchor="w").grid(
            row=row, column=0, sticky="w", **pad
        )

    if bbox is not None:
        label("Region:")
        tk.Label(
            root, text="{},{} {}×{}".format(*bbox), anchor="w"
        ).grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        row += 1

    label("Output file:")
    output_entry = tk.Entry(root, textvariable=output_var, width=42)
    output_entry.grid(row=row, column=1, sticky="we", **pad)

    def browse():
        path = filedialog.asksaveasfilename(
            defaultextension=".{}".format(fmt_var.get()),
            initialfile=os.path.basename(output_var.get()),
            initialdir=os.path.dirname(output_var.get()) or os.path.expanduser("~"),
            filetypes=[(fmt.upper(), "*.{}".format(fmt)) for fmt in _FORMATS],
        )
        if path:
            output_var.set(path)
            ext = os.path.splitext(path)[1].lstrip(".").lower()
            if ext in _FORMATS:
                fmt_var.set(ext)

    tk.Button(root, text="Browse…", command=browse).grid(
        row=row, column=2, sticky="w", **pad
    )
    row += 1

    label("Format:")
    fmt_menu = tk.OptionMenu(root, fmt_var, *_FORMATS)
    fmt_menu.grid(row=row, column=1, sticky="w", **pad)

    def on_fmt_change(*_args):
        # Keep the path's extension consistent with the dropdown so
        # the user isn't surprised by a .mp4 named "clip.gif".
        path = output_var.get()
        base, _ = os.path.splitext(path)
        if base:
            output_var.set("{}.{}".format(base, fmt_var.get()))

    fmt_var.trace_add("write", on_fmt_change)
    row += 1

    label("Title (first 3s):")
    tk.Entry(root, textvariable=title_var, width=42).grid(
        row=row, column=1, columnspan=2, sticky="we", **pad
    )
    row += 1

    label("Overlay text:")
    tk.Entry(root, textvariable=overlay_var, width=42).grid(
        row=row, column=1, columnspan=2, sticky="we", **pad
    )
    row += 1

    label("FPS:")
    tk.Spinbox(root, from_=1, to=120, textvariable=fps_var, width=6).grid(
        row=row, column=1, sticky="w", **pad
    )
    row += 1

    label("Duration (s):")
    tk.Entry(root, textvariable=duration_var, width=10).grid(
        row=row, column=1, sticky="w", **pad
    )
    tk.Label(
        root, text="(blank = until Ctrl-C)", fg="#666"
    ).grid(row=row, column=2, sticky="w", **pad)
    row += 1

    label("Start delay (s):")
    tk.Spinbox(
        root, from_=0, to=60, textvariable=countdown_var, width=6
    ).grid(row=row, column=1, sticky="w", **pad)
    tk.Label(
        root, text="(time to hide this window)", fg="#666"
    ).grid(row=row, column=2, sticky="w", **pad)
    row += 1

    tk.Checkbutton(
        root, text="Visualise mouse clicks", variable=show_clicks_var
    ).grid(row=row, column=1, columnspan=2, sticky="w", **pad)
    row += 1

    result = {"config": None}

    def on_record():
        try:
            duration = (
                float(duration_var.get())
                if duration_var.get().strip()
                else None
            )
        except ValueError:
            messagebox.showerror(
                "fastgrab", "Duration must be a number or blank"
            )
            return
        result["config"] = {
            "output": output_var.get(),
            "format": fmt_var.get(),
            "title": title_var.get() or None,
            "overlay_text": overlay_var.get() or None,
            "fps": int(fps_var.get()),
            "duration": duration,
            "countdown": int(countdown_var.get()),
            "show_clicks": bool(show_clicks_var.get()),
            "bbox": bbox,
        }
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
    tk.Button(btn_frame, text="Record", command=on_record, width=12).pack(
        side="right", padx=4
    )
    tk.Button(btn_frame, text="Cancel", command=on_cancel, width=12).pack(
        side="right", padx=4
    )

    root.bind("<Escape>", lambda _e: on_cancel())
    root.bind("<Return>", lambda _e: on_record())

    root.mainloop()
    return result["config"]
