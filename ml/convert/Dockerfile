# Conversion runs here, never on the harness (ADR 0006 C5). amd64: mmcv has no
# arm64 CPU wheels, so build with `docker build --platform linux/amd64` on Apple
# Silicon (runs under emulation there; native on amd64 CI/hosts).
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Pinned, in dependency order. torch/mmcv/mmdet/mmpose is the fragile chain ADR 0006
# warned about; these versions are the ones actually verified to import together.
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -U openmim==0.3.9 && \
    mim install --no-cache-dir mmengine==0.10.4 && \
    mim install --no-cache-dir "mmcv==2.1.0" && \
    mim install --no-cache-dir "mmdet==3.3.0"
# mmpose has no PyPI configs bundle for the rtmpose project configs; clone at a
# pinned tag and install editable so `mmpose::` config refs and projects/ resolve.
RUN git clone --depth 1 --branch v1.3.2 https://github.com/open-mmlab/mmpose.git /opt/mmpose && \
    pip install --no-cache-dir -r /opt/mmpose/requirements/build.txt && \
    pip install --no-cache-dir -r /opt/mmpose/requirements/runtime.txt && \
    pip install --no-cache-dir -e /opt/mmpose

RUN pip install --no-cache-dir coremltools==7.2 numpy==1.26.4 pillow==10.4.0 requests==2.32.3

WORKDIR /workspace
COPY convert.py /workspace/convert.py

ENTRYPOINT ["python", "/workspace/convert.py"]
