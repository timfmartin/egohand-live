#!/usr/bin/env bash
# Prepare the Isaac Sim side (isaac/sharpa_hands.py). Run on the machine that runs Isaac Sim.
# Idempotent. Does not modify the Isaac Sim install:
#   * fetches the official Sharpa Wave URDF/USD assets (pinned; not redistributed here)
#   * installs pyzmq + msgpack for Isaac Sim's Python into third_party/isaac_pydeps
# Isaac Sim 6 / Isaac Lab 3 already bundle torch, pinocchio and dex-retargeting.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACSIM_PATH="${ISAACSIM_PATH:-${HOME}/isaacsim}"
ISAAC_PY="${ISAAC_PY:-${ISAACSIM_PATH}/python.sh}"
# Same pin as IsaacCapture's examples/teleop_ros2/scripts/fetch_sharpa_wave_urdfs.py
SHARPA_URL="${SHARPA_URL:-https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git}"
SHARPA_REF="${SHARPA_REF:-3e953f588ba9954cebaa720aaa4cee06a43a068e}"
SHARPA_DIR="${REPO_ROOT}/third_party/sharpa-urdf-usd-xml"
PYDEPS="${REPO_ROOT}/third_party/isaac_pydeps"

log() { printf '\n\033[1;32m[setup-isaac]\033[0m %s\n' "$*"; }

[[ -x "${ISAAC_PY}" ]] || { echo "Isaac Sim python not found at ${ISAAC_PY} (set ISAACSIM_PATH)" >&2; exit 1; }

log "Sharpa Wave assets @ ${SHARPA_REF:0:12} (left/right hands only)"
if [[ ! -d "${SHARPA_DIR}/.git" ]]; then
  git clone -q --filter=blob:none --no-checkout "${SHARPA_URL}" "${SHARPA_DIR}"
fi
git -C "${SHARPA_DIR}" sparse-checkout set --no-cone /README.md /wave_01/left_sharpa_wave/ /wave_01/right_sharpa_wave/
if [[ "$(git -C "${SHARPA_DIR}" rev-parse HEAD 2>/dev/null)" != "${SHARPA_REF}" ]]; then
  git -C "${SHARPA_DIR}" fetch -q origin "${SHARPA_REF}" 2>/dev/null || true
  git -C "${SHARPA_DIR}" checkout -q --detach "${SHARPA_REF}"
fi

log "pyzmq + msgpack for Isaac Sim's Python -> ${PYDEPS}"
"${ISAAC_PY}" -m pip install -q --upgrade --target "${PYDEPS}" pyzmq msgpack

log "Check"
"${ISAAC_PY}" - "${REPO_ROOT}" <<'EOF'
import sys
repo = sys.argv[1]
sys.path[:0] = [repo, f"{repo}/third_party/isaac_pydeps"]
import zmq, msgpack  # noqa: F401
from egohand_live.retarget import HandRetargeter
for side in ("left", "right"):
    rt = HandRetargeter("sharpa_wave", side)
    usd = rt.spec.path("usd", side)
    assert usd.is_file(), usd
    print(f"  {side}: {len(rt.joint_names)} joints, USD {usd.relative_to(repo)}")
print("  zmq", zmq.__version__, "| ok")
EOF
log "Done. Run: ${ISAAC_PY} isaac/sharpa_hands.py --connect tcp://GPU_HOST:5555"
