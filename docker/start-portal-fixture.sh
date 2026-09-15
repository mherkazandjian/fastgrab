#!/bin/bash
# Bring up a private PipeWire graph carrying one synthetic video source, then
# run the test command against it.
#
# No compositor and no DRM device are involved. That is the point: a portal
# backend's frame-consumption half only needs a PipeWire node id, and PipeWire
# will carry CPU-backed buffers on a machine with no GPU at all — measured,
# three 320x240 BGRx frames read by a client with /dev/dri absent.
#
# Two things are easy to get wrong here and both look like "no frames":
#   * without a session manager nothing links the graph, so a consumer sits
#     reading nothing. WirePlumber is started for that and nothing else.
#   * pipewiresink defaults to media.class=Stream/Output/Video, which a
#     consumer will not treat as a source. It is overridden to Video/Source.
set -e

: "${XDG_RUNTIME_DIR:=/tmp/xdg-runtime}"
export XDG_RUNTIME_DIR
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

: "${FASTGRAB_FIXTURE_WIDTH:=320}"
: "${FASTGRAB_FIXTURE_HEIGHT:=240}"
: "${FASTGRAB_FIXTURE_NODE:=fastgrab-fixture}"
export FASTGRAB_FIXTURE_WIDTH FASTGRAB_FIXTURE_HEIGHT FASTGRAB_FIXTURE_NODE

pipewire >/tmp/pipewire.log 2>&1 &
PW_PID=$!
sleep 2
wireplumber >/tmp/wireplumber.log 2>&1 &
WP_PID=$!
sleep 2

# Solid red, so a consumer that mixes up the channel order is obvious:
# BGRx means byte 2 carries red.
# No long-lived synthetic source here on purpose. `pipewiresink
# mode=provide` is not a durable provider: the moment its consumer
# disconnects it errors with "all buffers have been removed / PipeWire
# link to remote node was destroyed" and exits, so a shared source
# survives exactly one test. Each test starts and stops its own; this
# script only guarantees a PipeWire graph with a session manager, which
# is the part that is slow and global.

cleanup() {
    kill "$WP_PID" "$PW_PID" 2>/dev/null || true
    wait "$WP_PID" "$PW_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the daemon itself; the video source is per-test now.
for _ in $(seq 1 60); do
    if pw-cli info 0 >/dev/null 2>&1; then
        break
    fi
    sleep 0.25
done

if ! pw-cli info 0 >/dev/null 2>&1; then
    echo "portal fixture: the PipeWire daemon never came up" >&2
    tail -5 /tmp/pipewire.log /tmp/wireplumber.log >&2 || true
    exit 1
fi

exec "$@"
