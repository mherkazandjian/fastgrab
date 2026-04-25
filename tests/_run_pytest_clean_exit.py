"""Wrapper that runs pytest and exits via ``os._exit`` to bypass Python's
normal interpreter shutdown.

Without this, the wlr integration suite trips a process-exit segfault
inside pywayland's cffi finalizers when the (already-disconnected)
Wayland display state is freed in an unfortunate order. Pytest itself
has already reported success at that point — the segfault only affects
the *exit code* the OS observes, which would otherwise turn green
test runs into red CI jobs.

``os._exit`` skips atexit + module finalisers, which is exactly what we
want here: pytest's ``main()`` has fully finished and there's no real
cleanup left to run.
"""
import os
import sys

import pytest


if __name__ == "__main__":
    code = pytest.main(sys.argv[1:])
    os._exit(int(code))
