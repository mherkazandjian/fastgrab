#!/bin/bash
set -e

: "${DISPLAY:=:99}"
: "${SCREEN_GEOMETRY:=1920x1080x24}"
: "${VNC_PORT:=5900}"

Xvfb "$DISPLAY" -screen 0 "$SCREEN_GEOMETRY" -ac +extension GLX +render -noreset \
    >/tmp/xvfb.log 2>&1 &

for _ in $(seq 1 50); do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done

fluxbox >/tmp/fluxbox.log 2>&1 &

# A visible terminal so there's something to capture in the virtual desktop.
xterm -geometry 100x30+40+40 -title "fastgrab" \
      -e "bash -c 'echo fastgrab dev desktop on $DISPLAY $SCREEN_GEOMETRY; exec bash'" \
      >/dev/null 2>&1 &

if [ "$#" -eq 0 ]; then
    # No command given: run x11vnc in foreground so the container stays alive
    # while the desktop is reachable on $VNC_PORT.
    exec x11vnc -display "$DISPLAY" -forever -shared -rfbport "$VNC_PORT" \
                -nopw -quiet
else
    x11vnc -display "$DISPLAY" -forever -shared -rfbport "$VNC_PORT" \
           -nopw -quiet -bg -o /tmp/x11vnc.log
    exec "$@"
fi
