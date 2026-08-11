"""Flameshot-style frozen-screenshot region selector.

On entry the selector grabs a snapshot of the entire virtual desktop
via :class:`fastgrab.screenshot.Screenshot`, shows a darkened copy in a
borderless fullscreen Tk window, and lets the user drag a rectangle.
The inside of the rectangle is redrawn at full brightness so the
selection visually "uncovers" the live screen content underneath the
dim mask. Returns ``(x, y, w, h)`` in screen-absolute pixels, or
``None`` on Escape / trivial click.
"""


# Brightness multiplier for the dim mask. 0.4 keeps enough detail
# visible to aim while clearly distinguishing "frozen + dimmed" from
# the bright cut-out of the selection.
DIM_FACTOR = 0.4

# Drag-redraw throttling. At 7040×1440 each redraw crops + uploads a
# PhotoImage; on a fast machine that's ~5ms but on busy systems it can
# stutter. Coalescing drag events to ~50 fps keeps the rectangle smooth
# without spamming the event loop.
REDRAW_INTERVAL_MS = 20


def select_region():
    """Open the selector. Returns ``(x, y, w, h)`` or ``None``."""
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError(
            "fastgrab-record --gui needs tkinter — install your "
            "distro's python3-tk package"
        ) from exc
    try:
        from PIL import Image, ImageTk
    except ImportError as exc:
        raise RuntimeError(
            "fastgrab-record --gui needs Pillow — install with "
            "pip install fastgrab[gui]"
        ) from exc
    import numpy

    from fastgrab.screenshot import Screenshot

    # 1. Capture the whole virtual desktop. This is the "frozen" frame
    #    we'll display while the user picks a region. The capture copy
    #    is ours — Screenshot's internal buffer gets reused on the next
    #    capture(), so we np.array() it to detach.
    grab = Screenshot()
    bgra = numpy.array(grab.capture(), copy=True)
    h, w = bgra.shape[:2]

    # 2. BGRA → RGB for Pillow. The captured alpha channel is zeroed
    #    on most X11 setups, so dropping it is safe.
    rgb = numpy.ascontiguousarray(bgra[..., [2, 1, 0]])
    bright = Image.fromarray(rgb, mode="RGB")

    # 3. Dim copy. Done in numpy because Pillow's ImageEnhance allocates
    #    an extra full-size buffer; numpy multiply-then-astype is cheaper.
    dark_arr = (rgb.astype(numpy.uint16) * int(DIM_FACTOR * 256) // 256)
    dark = Image.fromarray(dark_arr.astype(numpy.uint8), mode="RGB")

    root = tk.Tk()
    root.title("fastgrab — select region")
    # overrideredirect drops WM decoration and stops the WM from
    # constraining us to one monitor; the explicit geometry below
    # then forces the window to span the entire X screen.
    root.overrideredirect(True)
    root.geometry("{}x{}+0+0".format(w, h))
    root.attributes("-topmost", True)
    root.configure(bg="black")

    canvas = tk.Canvas(
        root, width=w, height=h,
        highlightthickness=0, bg="black", cursor="crosshair",
    )
    canvas.pack(fill="both", expand=True)

    dark_tk = ImageTk.PhotoImage(dark)
    canvas.create_image(0, 0, image=dark_tk, anchor="nw")
    # Tk's image GC drops PhotoImages whose only reference is in the
    # canvas item list — pin the dark mask on an attribute so it lives
    # as long as the canvas.
    canvas._dark_tk = dark_tk

    state = {
        "x0": None, "y0": None,
        "rect_id": None,
        "selection_image_id": None,
        "selection_tk": None,
        "pending": None,
        "pending_args": None,
        "result": None,
    }

    def _redraw_selection(x, y, w_sel, h_sel):
        # Clamp to screen so a drag past the edge doesn't index out of
        # the bright PIL image.
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        w_sel = max(1, min(w_sel, w - x))
        h_sel = max(1, min(h_sel, h - y))

        cropped = bright.crop((x, y, x + w_sel, y + h_sel))
        cropped_tk = ImageTk.PhotoImage(cropped)
        # Hold the new ref before deleting the old to dodge a flicker
        # window where neither image is on the canvas.
        if state["selection_image_id"] is not None:
            canvas.delete(state["selection_image_id"])
        state["selection_image_id"] = canvas.create_image(
            x, y, image=cropped_tk, anchor="nw",
        )
        state["selection_tk"] = cropped_tk

        if state["rect_id"] is not None:
            canvas.delete(state["rect_id"])
        state["rect_id"] = canvas.create_rectangle(
            x, y, x + w_sel, y + h_sel,
            outline="#ff5050", width=2,
        )

    def _flush_pending():
        state["pending"] = None
        if state["pending_args"] is not None:
            args = state["pending_args"]
            state["pending_args"] = None
            _redraw_selection(*args)

    def _schedule(x, y, w_sel, h_sel):
        # Coalesce consecutive drag events: store the latest bounds and
        # let a single delayed callback do the work. This caps redraw
        # rate at ~50 fps even if Motion events arrive faster.
        state["pending_args"] = (x, y, w_sel, h_sel)
        if state["pending"] is None:
            state["pending"] = root.after(REDRAW_INTERVAL_MS, _flush_pending)

    def on_press(e):
        state["x0"] = e.x
        state["y0"] = e.y

    def on_drag(e):
        if state["x0"] is None:
            return
        x = min(state["x0"], e.x)
        y = min(state["y0"], e.y)
        w_sel = abs(e.x - state["x0"])
        h_sel = abs(e.y - state["y0"])
        if w_sel < 1 or h_sel < 1:
            return
        _schedule(x, y, w_sel, h_sel)

    def on_release(e):
        if state["x0"] is None:
            root.destroy()
            return
        x = min(state["x0"], e.x)
        y = min(state["y0"], e.y)
        w_sel = abs(e.x - state["x0"])
        h_sel = abs(e.y - state["y0"])
        # Reject trivial click-without-drag — too tiny to be a useful
        # recording region and usually means "I changed my mind".
        if w_sel >= 8 and h_sel >= 8:
            state["result"] = (x, y, w_sel, h_sel)
        root.destroy()

    def on_escape(_e):
        root.destroy()

    canvas.bind("<Button-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_escape)

    # overrideredirect windows don't get focus from the WM; force it so
    # Escape and the mouse bindings actually receive events.
    root.focus_force()

    root.mainloop()
    return state["result"]
