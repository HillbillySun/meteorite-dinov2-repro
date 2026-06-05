#!/usr/bin/env bash
set -euo pipefail

# Expose CUDA/cuDNN runtime libraries installed by pip packages such as
# nvidia-cudnn-cu12 and nvidia-cublas-cu12, so onnxruntime-gpu can load CUDAExecutionProvider.
export LD_LIBRARY_PATH="$(python - <<'PY'
import os
import site
paths = []
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    for sub in [
        "nvidia/cudnn/lib",
        "nvidia/cublas/lib",
        "nvidia/cuda_runtime/lib",
    ]:
        path = os.path.join(sp, sub)
        if os.path.isdir(path):
            paths.append(path)
print(":".join(paths))
PY
):${LD_LIBRARY_PATH:-}"

python - <<'PY'
import torch
import onnxruntime as ort
print("torch cuda available:", torch.cuda.is_available())
print("onnxruntime providers:", ort.get_available_providers())
PY
