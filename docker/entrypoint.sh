#!/bin/bash
set -e

# Build the C extension in-place against the container's Python/X11 so the
# fastgrab package is importable from the mounted source tree. The resulting
# .so files are gitignored, so this won't show up in `git status` on the host.
if [ -f pyproject.toml ]; then
    pip install --quiet --no-build-isolation -e . >/dev/null
fi

exec "$@"
