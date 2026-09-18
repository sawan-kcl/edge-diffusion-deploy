# Edge-sim runtime for SANA. Base image ships CUDA + cuDNN + TensorRT + PyTorch,
# so we only add the Python deps.
#
# Build:  docker build -t edge-diffusion -f docker/Dockerfile .
# Run:    docker run --gpus all -it --rm -v "$PWD":/work -w /work edge-diffusion
#
# (You can also skip the build and just run the NGC base directly, mounting this
#  folder and `pip install -r requirements.txt` inside.)

FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /work
COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -r /work/requirements.txt

# Weights download into ./models (mounted at /work) instead of the shared host
# ~/.cache/huggingface, so nothing lands outside the project folder.
ENV HF_HOME=/work/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# default: drop into a shell; override with e.g. `python src/pipeline.py ...`
CMD ["/bin/bash"]
