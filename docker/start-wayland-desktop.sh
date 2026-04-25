#!/bin/bash
# Boot a headless wlroots-based compositor (cage) and run the test command
# inside it. Cage hosts our test-only painter as the kiosk client; the
# painter advertises itself on $XDG_RUNTIME_DIR/fastgrab-painter.sock.
#
# Used by the `test-wayland` docker compose service.
set -e

: "${XDG_RUNTIME_DIR:=/tmp/xdg-runtime}"
export XDG_RUNTIME_DIR

mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Headless wlroots: no DRM/KMS device, no real input devices.
export WLR_BACKENDS=headless
export WLR_LIBINPUT_NO_DEVICES=1

# Cage names the wayland socket itself (typically wayland-0 / wayland-1
# whichever is free). Unset any inherited WAYLAND_DISPLAY so cage's choice
# wins, then detect the socket cage actually created.
unset WAYLAND_DISPLAY

# Start cage with the painter as the kiosk child.
cage -s -- python /app/tests/wayland_painter.py >/tmp/cage.log 2>&1 &
CAGE_PID=$!

# Wait for cage to create a wayland-N socket in XDG_RUNTIME_DIR.
WAYLAND_DISPLAY=
for _ in $(seq 1 100); do
    for sock in "$XDG_RUNTIME_DIR"/wayland-*; do
        case "$sock" in
            *.lock) ;;
            *)
                if [ -S "$sock" ]; then
                    WAYLAND_DISPLAY=$(basename "$sock")
                    break
                fi
                ;;
        esac
    done
    if [ -n "$WAYLAND_DISPLAY" ] \
       && [ -S "$XDG_RUNTIME_DIR/fastgrab-painter.sock" ]; then
        break
    fi
    sleep 0.1
done

if [ -z "$WAYLAND_DISPLAY" ]; then
    echo "no wayland-* socket appeared in $XDG_RUNTIME_DIR" >&2
    echo "--- runtime dir ---" >&2
    ls -la "$XDG_RUNTIME_DIR" >&2 || true
    echo "--- cage log ---" >&2
    cat /tmp/cage.log >&2 || true
    kill "$CAGE_PID" 2>/dev/null || true
    exit 1
fi
if [ ! -S "$XDG_RUNTIME_DIR/fastgrab-painter.sock" ]; then
    echo "painter socket never appeared" >&2
    echo "--- cage log ---" >&2
    cat /tmp/cage.log >&2 || true
    kill "$CAGE_PID" 2>/dev/null || true
    exit 1
fi
export WAYLAND_DISPLAY
echo "wayland ready: $WAYLAND_DISPLAY (cage pid=$CAGE_PID)"

# Hand off to the test command. Cage stays in the background; cleanup is
# the container's responsibility (--rm + tini as PID 1).
exec "$@"
