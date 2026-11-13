#!/usr/bin/env bash
# Install NVIDIA EgoPost (pinned) and egohand-live into ./.venv, then fetch model assets.
# Idempotent: safe to re-run. Requires uv, git, an NVIDIA driver with CUDA 13 support.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACCAPTURE_URL="${ISAACCAPTURE_URL:-https://github.com/NVIDIA/IsaacCapture.git}"
ISAACCAPTURE_REF="${ISAACCAPTURE_REF:-16578fe0ec0d9f2cf645c3f86d2380b9f0f2b3d7}"
UPSTREAM="${REPO_ROOT}/third_party/IsaacCapture"
EGOPOST="${UPSTREAM}/src/python/isaaccapture_egopost"
VENV="${REPO_ROOT}/.venv"
PY="${VENV}/bin/python"
UV="${UV:-$(command -v uv || echo "${HOME}/.local/bin/uv")}"

log() { printf '\n\033[1;32m[setup]\033[0m %s\n' "$*"; }

log "Upstream IsaacCapture @ ${ISAACCAPTURE_REF:0:12}"
if [[ ! -d "${UPSTREAM}/.git" ]]; then
  # LFS only holds upstream docs media; skip it.
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${ISAACCAPTURE_URL}" "${UPSTREAM}"
fi
if [[ "$(git -C "${UPSTREAM}" rev-parse HEAD)" != "${ISAACCAPTURE_REF}" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git -C "${UPSTREAM}" fetch --depth 1 origin "${ISAACCAPTURE_REF}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${UPSTREAM}" checkout --detach "${ISAACCAPTURE_REF}"
fi

log "Python venv (${VENV})"
[[ -x "${PY}" ]] || "${UV}" venv "${VENV}" --python 3.12
"${UV}" pip install --python "${PY}" torch torchvision
"${UV}" pip install --python "${PY}" -r "${EGOPOST}/requirements.txt"
"${UV}" pip install --python "${PY}" --no-deps -r "${EGOPOST}/requirements-dinov3.txt"
# onnxruntime-gpu's CUDA provider links libcurand.so.10, which the CUDA 13 torch wheels
# no longer pull in; without it ORT silently falls back to CPU (~10x slower detector).
"${UV}" pip install --python "${PY}" nvidia-curand
"${UV}" pip install --python "${PY}" -e "${REPO_ROOT}"

# EgoPost is not a pip package; its top-level modules (`egohand`, `inference`) are
# imported from the source tree. A .pth file puts it on sys.path for this venv only.
SITE="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${EGOPOST}" > "${SITE}/egopost_upstream.pth"

log "Model assets (Hugging Face cache)"
"${PY}" - <<'EOF'
from huggingface_hub import hf_hub_download
from soma import get_assets_dir
print("EgoHand ckpt :", hf_hub_download("nvidia/EgoHand-v1", "egohand_s060000.ckpt"))
print("RT-DETR onnx :", hf_hub_download("nvidia/RT-DETR-Hand-Detector-v1", "hand_detector.onnx"))
print("SOMA-X assets:", get_assets_dir())
EOF

log "Sanity check"
"${PY}" - <<'EOF'
import torch, onnxruntime as ort
from dinov3.models.vision_transformer import DinoVisionTransformer  # noqa: F401
import inference.pipeline.predictor  # noqa: F401  (EgoPost on sys.path)
print("torch", torch.__version__, "cuda", torch.version.cuda, "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("onnxruntime providers:", ort.get_available_providers())
EOF
log "Done. Activate with: source ${VENV}/bin/activate"
