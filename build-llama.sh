#!/usr/bin/env bash
# aios-scripts-version: 6
# Build llama.cpp from source on Ubuntu (22.04 / 24.04 / 26.04), tuned to the
# detected CPU / RAM / GPU(s).
#
#   1. Detects CPU, RAM and NVIDIA GPUs (count, VRAM, compute capability).
#   2. Installs missing toolchain pieces via apt: git, cmake, ninja, build-essential,
#      libcurl, and the NVIDIA CUDA Toolkit (>= 12.8 required for RTX 50 / sm_120).
#   3. Clones / updates llama.cpp (full clone - a shallow clone breaks `git pull`).
#   4. Configures with the CUDA arch list matching the installed cards and builds.
#   5. Installs binaries to $INSTALL_DIR and writes build-info.json there.
#
# Usage:  ./build-llama.sh [--clean] [--skip-install] [--branch <ref>] [--backend auto|cuda|vulkan|sycl|cpu]
#         sycl = Intel oneAPI backend for Intel Arc (docs/backend/SYCL.md); needs /opt/intel/oneapi (setvars.sh)
# Env:    SRC_DIR (default ./llama.cpp)   INSTALL_DIR (default ./bin)   LLAMA_BACKEND (same as --backend)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$SCRIPT_DIR/llama.cpp}"
INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR/bin}"
BRANCH="master"
BACKEND="${LLAMA_BACKEND:-auto}"
CLEAN=0
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --branch) BRANCH="$2"; shift ;;
    --backend) BACKEND="$2"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[1;33m! %s\033[0m\n' "$*"; }
SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"

# ---------------------------------------------------------------------------
# 1. Hardware detection
# ---------------------------------------------------------------------------
step "Detecting hardware"
CPU_NAME="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
PHYS_CORES="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
LOG_CORES="$(nproc)"
RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024 + 0.5}' /proc/meminfo)"
info "CPU : $CPU_NAME  ($PHYS_CORES cores / $LOG_CORES threads)"
info "RAM : ${RAM_GB} GB"

GPU_COUNT=0; CUDA_ARCHS=""; NEEDS_CUDA128=0; GPU_JSON="[]"
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits 2>/dev/null || true)
  GPU_COUNT=${#GPU_ROWS[@]}
  declare -A ARCHSET=()
  GPU_JSON="["
  for row in "${GPU_ROWS[@]}"; do
    IFS=',' read -r idx name vram cc drv <<<"$row"
    name="${name## }"; vram="${vram## }"; cc="${cc## }"; drv="${drv## }"
    info "GPU$idx: $name  $(( vram / 1024 )) GB  cc=$cc  driver=$drv"
    arch="${cc/./}"
    if [[ "$arch" =~ ^[0-9]{2,3}$ ]]; then ARCHSET[$arch]=1
    elif [[ "$name" == *"RTX 50"* ]]; then ARCHSET[120]=1
    elif [[ "$name" == *"RTX 40"* ]]; then ARCHSET[89]=1
    elif [[ "$name" == *"RTX 30"* ]]; then ARCHSET[86]=1
    fi
    [[ "$GPU_JSON" != "[" ]] && GPU_JSON+=","
    GPU_JSON+="{\"index\":$idx,\"name\":\"$name\",\"vram_mb\":$vram,\"compute_cap\":\"$cc\",\"driver\":\"$drv\"}"
  done
  GPU_JSON+="]"
  if [[ $GPU_COUNT -gt 0 && ${#ARCHSET[@]} -eq 0 ]]; then ARCHSET[86]=1; ARCHSET[89]=1; ARCHSET[120]=1; fi
  CUDA_ARCHS="$(printf '%s\n' "${!ARCHSET[@]}" | sort -n | paste -sd';')"
  [[ -n "${ARCHSET[120]:-}" ]] && NEEDS_CUDA128=1
  [[ -n "$CUDA_ARCHS" ]] && info "CUDA architectures: $CUDA_ARCHS"
else
  warn "nvidia-smi not found. Install the NVIDIA driver first (e.g. 'sudo ubuntu-drivers install' or nvidia-driver-580) if this box has GPUs."
fi
OTHER_GPUS="$(lspci 2>/dev/null | grep -iE 'vga|3d|display' | grep -viE 'nvidia' | sed 's/.*: //' | paste -sd';' || true)"
[[ -n "$OTHER_GPUS" ]] && info "Other GPUs: $OTHER_GPUS"
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
if [[ "$BACKEND" == auto ]]; then
  if (( GPU_COUNT > 0 )); then BACKEND=cuda
  elif [[ "$OTHER_GPUS" == *Intel* && "$OTHER_GPUS" == *Arc* && -f "$ONEAPI_ROOT/setvars.sh" ]]; then BACKEND=sycl
  elif [[ -n "$OTHER_GPUS" ]]; then BACKEND=vulkan
  else BACKEND=cpu; fi
fi
[[ "$BACKEND" == cuda && $GPU_COUNT -eq 0 ]] && { warn "Backend cuda requested but no NVIDIA GPU found - building anyway."; [[ -z "$CUDA_ARCHS" ]] && CUDA_ARCHS="86;89;120"; }
info "Backend: $BACKEND"

# ---------------------------------------------------------------------------
# 2. Toolchain
# ---------------------------------------------------------------------------
step "Checking toolchain"
apt_install() {
  if [[ $SKIP_INSTALL -eq 1 ]]; then warn "Missing packages: $* (--skip-install set)"; return; fi
  info "apt install $*"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "$@"
}
PKGS=()
command -v git   >/dev/null || PKGS+=(git)
command -v cmake >/dev/null || PKGS+=(cmake)
command -v ninja >/dev/null || PKGS+=(ninja-build)
command -v g++   >/dev/null || PKGS+=(build-essential)
dpkg -s libssl-dev >/dev/null 2>&1 || PKGS+=(libssl-dev)
dpkg -s pkg-config >/dev/null 2>&1 || PKGS+=(pkg-config)
[[ ${#PKGS[@]} -gt 0 ]] && apt_install "${PKGS[@]}"

# cmake >= 3.24 needed; Ubuntu 22.04 ships 3.22 -> use Kitware's repo
CMAKE_VER="$(cmake --version | head -1 | awk '{print $3}')"
if [[ "$(printf '%s\n3.24\n' "$CMAKE_VER" | sort -V | head -1)" != "3.24" ]]; then
  warn "cmake $CMAKE_VER is too old; installing from Kitware apt repo"
  if [[ $SKIP_INSTALL -eq 0 ]]; then
    apt_install ca-certificates gpg wget
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor | $SUDO tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(. /etc/os-release; echo "$UBUNTU_CODENAME") main" | $SUDO tee /etc/apt/sources.list.d/kitware.list >/dev/null
    apt_install cmake
  fi
fi

# CUDA toolkit
CUDA_ROOT=""; CUDA_VER=""
find_cuda() {
  local c
  for c in "${CUDA_PATH:-}" /usr/local/cuda /usr/local/cuda-13* /usr/local/cuda-12*; do
    [[ -x "$c/bin/nvcc" ]] && { echo "$c"; return; }
  done
  command -v nvcc >/dev/null && dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")"
}
cuda_ver_of() { "$1/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+'; }
ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }

if [[ "$BACKEND" == cuda ]]; then
  CUDA_ROOT="$(find_cuda || true)"
  [[ -n "$CUDA_ROOT" ]] && CUDA_VER="$(cuda_ver_of "$CUDA_ROOT")"
  if [[ -z "$CUDA_ROOT" ]] || { [[ $NEEDS_CUDA128 -eq 1 ]] && ! ver_ge "$CUDA_VER" 12.8; }; then
    [[ -n "$CUDA_ROOT" ]] && warn "CUDA $CUDA_VER is too old for Blackwell (need >= 12.8)"
    if [[ $SKIP_INSTALL -eq 0 ]]; then
      . /etc/os-release
      DISTRO="ubuntu${VERSION_ID/./}"           # ubuntu2204 / ubuntu2404 / ubuntu2604
      KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
      info "Adding NVIDIA CUDA apt repo for $DISTRO"
      wget -qO /tmp/cuda-keyring.deb "$KEYRING_URL" || { warn "No NVIDIA repo for $DISTRO; falling back to ubuntu2404"; wget -qO /tmp/cuda-keyring.deb "${KEYRING_URL/$DISTRO/ubuntu2404}"; }
      $SUDO dpkg -i /tmp/cuda-keyring.deb
      # cuda-toolkit (no driver) - the driver is managed separately (nvidia-driver-*)
      apt_install cuda-toolkit
      CUDA_ROOT="$(find_cuda)"; CUDA_VER="$(cuda_ver_of "$CUDA_ROOT")"
    else
      echo "CUDA toolkit >= 12.8 required." >&2; exit 1
    fi
  fi
  info "CUDA : $CUDA_VER  ($CUDA_ROOT)"
  export CUDA_PATH="$CUDA_ROOT" CUDACXX="$CUDA_ROOT/bin/nvcc" PATH="$CUDA_ROOT/bin:$PATH"

  # nvcc only supports host GCC up to a certain major; pick a supported one if needed
  GCC_MAJOR="$(gcc -dumpfullversion -dumpversion | cut -d. -f1)"
  MAX_GCC=14; ver_ge "$CUDA_VER" 13.0 && MAX_GCC=15
  HOST_CXX="$(command -v g++)"
  if (( GCC_MAJOR > MAX_GCC )); then
    for v in $MAX_GCC $((MAX_GCC-1)) $((MAX_GCC-2)); do
      if command -v "g++-$v" >/dev/null; then HOST_CXX="$(command -v g++-$v)"; break; fi
    done
    if [[ "$HOST_CXX" == "$(command -v g++)" ]]; then
      warn "gcc $GCC_MAJOR is newer than CUDA $CUDA_VER supports; installing g++-$MAX_GCC as CUDA host compiler"
      apt_install "gcc-$MAX_GCC" "g++-$MAX_GCC" && HOST_CXX="$(command -v g++-$MAX_GCC)"
    fi
  fi
  info "CUDA host compiler: $HOST_CXX"
fi

if [[ "$BACKEND" == sycl ]]; then
  # Intel oneAPI Base Toolkit (icx/icpx + oneMKL + oneDNN) - see llama.cpp docs/backend/SYCL.md "Linux"
  if [[ ! -f "$ONEAPI_ROOT/setvars.sh" ]]; then
    if [[ $SKIP_INSTALL -eq 1 ]]; then warn "Missing: Intel oneAPI ($ONEAPI_ROOT) (SkipInstall set)"; else
      info "Installing Intel oneAPI Base Toolkit from Intel's apt repo (several GB) ..."
      wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
      echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
      sudo apt-get update -qq && apt_install intel-oneapi-base-toolkit
    fi
  fi
  [[ -f "$ONEAPI_ROOT/setvars.sh" ]] || { echo "Intel oneAPI not found at $ONEAPI_ROOT - install the Base Toolkit and re-run." >&2; exit 1; }
  set +u; source "$ONEAPI_ROOT/setvars.sh" >/dev/null 2>&1; set -u
  command -v icpx >/dev/null || { echo "icpx not on PATH after sourcing $ONEAPI_ROOT/setvars.sh" >&2; exit 1; }
  info "oneAPI: $ONEAPI_ROOT  ($(icpx --version 2>/dev/null | head -1))"
  # llama.cpp's SYCL kernels reference intel_gpu_bmg_g31 (Arc Pro B60/B70 die) - needs DPC++ >= 2025.2
  ICPX_VER="$(icpx --version 2>/dev/null | grep -oE 'Version [0-9]+\.[0-9]+' | head -1 | cut -d' ' -f2)"
  if [[ -n "$ICPX_VER" ]] && (( $(echo "$ICPX_VER" | cut -d. -f1) < 2025 || ( $(echo "$ICPX_VER" | cut -d. -f1) == 2025 && $(echo "$ICPX_VER" | cut -d. -f2) < 2 ) )); then
    echo "DPC++ $ICPX_VER is too old for the Arc B-series SYCL kernels (need >= 2025.2): sudo apt-get install intel-oneapi-base-toolkit (or intel-deep-learning-essentials) to get the current release, then re-run." >&2; exit 1
  fi
  info "sycl-ls: $(sycl-ls 2>/dev/null | grep -i 'level_zero:gpu' | paste -sd';' || echo 'no level_zero GPU - install the Intel GPU driver and add yourself to the render/video groups')"
fi

if [[ "$BACKEND" == vulkan ]]; then
  VPKGS=()
  dpkg -s libvulkan-dev >/dev/null 2>&1 || VPKGS+=(libvulkan-dev)
  command -v glslc >/dev/null || VPKGS+=(glslc)
  dpkg -s vulkan-tools >/dev/null 2>&1 || VPKGS+=(vulkan-tools)
  [[ ${#VPKGS[@]} -gt 0 ]] && apt_install "${VPKGS[@]}"
  info "Vulkan: $(vulkaninfo --summary 2>/dev/null | grep -m1 -i 'deviceName' || echo 'vulkaninfo unavailable')"
fi

# ---------------------------------------------------------------------------
# 3. Source
# ---------------------------------------------------------------------------
step "Fetching llama.cpp source"
if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$SRC_DIR"
fi
pushd "$SRC_DIR" >/dev/null
[[ -f .git/shallow ]] && { info "Un-shallowing clone"; git fetch --unshallow; }
git fetch origin
git checkout -q "$BRANCH"
git reset -q --hard "origin/$BRANCH" 2>/dev/null || true
COMMIT="$(git rev-parse --short HEAD)"
TAG="$(git describe --tags --always 2>/dev/null)"
info "llama.cpp $TAG ($COMMIT)"
popd >/dev/null

# ---------------------------------------------------------------------------
# 4. Configure + build
# ---------------------------------------------------------------------------
step "Configuring"
BUILD_DIR="$SRC_DIR/build"
[[ $CLEAN -eq 1 ]] && rm -rf "$BUILD_DIR"

CMAKE_ARGS=(
  -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=$([[ "$BACKEND" == sycl ]] && echo ON || echo OFF)   # static (SYCL: shared, as in examples/sycl/build.sh)
  -DGGML_NATIVE=ON                   # -march=native (Zen5: AVX-512)
  -DLLAMA_OPENSSL=ON                 # HTTPS for -hf model downloads (needs libssl-dev)
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=ON
  -DLLAMA_BUILD_SERVER=ON
)
[[ "$BACKEND" == vulkan ]] && CMAKE_ARGS+=(-DGGML_VULKAN=ON)
# docs/backend/SYCL.md "Linux": icx for C, icpx for C++, FP16 on
[[ "$BACKEND" == sycl ]] && CMAKE_ARGS+=(-DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx)
if [[ -n "$CUDA_ROOT" ]]; then
  CMAKE_ARGS+=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc"
    -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX"
    -DGGML_CUDA_FA_ALL_QUANTS=ON     # flash-attn kernels for q4_0/q8_0 KV cache
    -DGGML_CUDA_GRAPHS=ON
  )
fi
cmake "${CMAKE_ARGS[@]}"

step "Building with $LOG_CORES threads (CUDA kernels take a while)"
cmake --build "$BUILD_DIR" --config Release -j "$LOG_CORES"

# ---------------------------------------------------------------------------
# 5. Install
# ---------------------------------------------------------------------------
step "Installing to $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
find "$BUILD_DIR/bin" -maxdepth 1 -type f -name 'llama-*' -perm -u+x -exec cp -f {} "$INSTALL_DIR/" \;
find "$BUILD_DIR/bin" -maxdepth 1 -type f -name '*.so*' -exec cp -f {} "$INSTALL_DIR/" \; 2>/dev/null || true

cat >"$INSTALL_DIR/build-info.json" <<EOF
{
  "built_at": "$(date -Is)",
  "backend": "$BACKEND",
  "oneapi_root": "$([[ "$BACKEND" == sycl ]] && echo "$ONEAPI_ROOT")",
  "llama_tag": "$TAG",
  "llama_commit": "$COMMIT",
  "cuda": "$CUDA_VER",
  "cuda_archs": "$CUDA_ARCHS",
  "cpu": "$CPU_NAME",
  "cores": $PHYS_CORES,
  "ram_gb": $RAM_GB,
  "gpus": $GPU_JSON
}
EOF

step "Smoke test"
"$INSTALL_DIR/llama-cli" --version || true
[[ "$BACKEND" == sycl && -x "$INSTALL_DIR/llama-ls-sycl-device" ]] && "$INSTALL_DIR/llama-ls-sycl-device" 2>&1 || true
[[ "$BACKEND" != cpu ]] && "$INSTALL_DIR/llama-server" --list-devices 2>&1 || true
[[ "$BACKEND" == sycl ]] && warn "SYCL binaries need the oneAPI runtime: 'source $ONEAPI_ROOT/setvars.sh' before launching llama-server by hand (AIOS does it for you)." || true

printf '\n\033[1;32mDone. Binaries in %s\033[0m\n' "$INSTALL_DIR"
printf '\033[1;32mNext: ./run-qwen.sh to download + launch Qwen3.8-27B tuned for this machine.\033[0m\n'
