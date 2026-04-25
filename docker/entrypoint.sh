#!/bin/bash
set -e

# Build the C extension in-place against the container's Python/X11 so the
# fastgrab package is importable from the mounted source tree. The resulting
# .so files are gitignored, so this won't show up in `git status` on the host.
if [ -f pyproject.toml ]; then
    pip install --quiet --no-build-isolation -e . >/dev/null
    # poetry-core's editable install does not place compiled extensions
    # next to the source — do that ourselves via build.py so
    # `import fastgrab._linux_x11` works.
    python build.py >/dev/null
fi

exec "$@"
