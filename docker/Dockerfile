FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:99 \
    SCREEN_GEOMETRY=1920x1080x24 \
    VNC_PORT=5900

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libx11-dev \
        libgomp1 \
        xvfb \
        xauth \
        x11vnc \
        fluxbox \
        xterm \
        x11-apps \
        x11-utils \
        x11-xserver-utils \
        xdotool \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry numpy pytest python-xlib \
    && poetry config virtualenvs.create false

WORKDIR /app

COPY docker/entrypoint.sh /usr/local/bin/entrypoint
COPY docker/start-desktop.sh /usr/local/bin/start-desktop
RUN chmod +x /usr/local/bin/entrypoint /usr/local/bin/start-desktop

EXPOSE 5900

ENTRYPOINT ["entrypoint"]
CMD ["bash"]
