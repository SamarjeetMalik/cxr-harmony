# Runtime image for the pipeline.
#
# Two-stage so the build toolchain does not ship. opencv-python-headless is used
# rather than opencv-python: the GUI build pulls in an X11 stack that a data
# pipeline never uses and that adds several hundred megabytes to an image which
# will run on a server with no display.
FROM python:3.11-slim AS build

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir .


FROM python:3.11-slim

# libgl and libglib are the runtime shared objects OpenCV links against even in
# the headless build; without them `import cv2` fails at load time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The pipeline handles identifiable data on its way in, so it does not run as
# root and the working directory is owned by the unprivileged user.
RUN useradd --create-home --uid 10001 pipeline
WORKDIR /data
RUN chown pipeline:pipeline /data
USER pipeline

COPY --chown=pipeline:pipeline configs /opt/cxr-harmony/configs

ENTRYPOINT ["cxr-harmony"]
CMD ["--help"]
