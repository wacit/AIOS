#!/usr/bin/env python3
"""
AIOS - Browser OS for llama.cpp
================================
Single-file, dependency-free (Python 3.9+ stdlib only) local web app that:

  * detects the machine (OS, CPU, RAM, NVIDIA / Intel / AMD GPUs, toolchain)
  * builds llama.cpp on the fly for that hardware by driving build-llama.ps1 / build-llama.sh
  * recommends a model quant that fits the detected VRAM and downloads it from Hugging Face
  * starts / stops llama-server with tuned flags and lets you test it from the browser

Run:   python aios.py            (then open http://localhost:7800)
       python aios.py --port 7800 --host 0.0.0.0

Files live next to this script:  build-llama.ps1 / build-llama.sh, bin/, models/, aios-state.json
"""
import argparse
import tempfile
import ctypes
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP_VERSION = "0.4.2"
SCRIPTS_VERSION = 6   # bump when the embedded build/launch scripts change
ROOT = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(ROOT, "bin")
MODEL_DIR = os.environ.get("AIOS_MODELS") or os.path.join(ROOT, "models")   # overridable: --models-dir, GUI, AIOS_MODELS
STATE_FILE = os.path.join(ROOT, "aios-state.json")
IS_WIN = platform.system() == "Windows"
EXE = ".exe" if IS_WIN else ""
HF_UA = f"aios/{APP_VERSION} (llama.cpp builder)"
HF_BASE = os.environ.get("AIOS_HF_BASE", "https://huggingface.co")
GH_API = os.environ.get("AIOS_GH_BASE", "https://api.github.com")
LLAMA_REPO = "ggml-org/llama.cpp"

# ---------------------------------------------------------------------------
# Model catalog. Only families whose GGUF repos are well established; the app
# always fetches the real file list from Hugging Face before recommending.
# ---------------------------------------------------------------------------
CATALOG = [
    {"id": "qwen3.8-27b", "name": "Qwen3.8-27B", "repo": "unsloth/Qwen3.8-27B-GGUF",
     "params_b": 27, "arch": "hybrid (48 DeltaNet + 16 attention)", "kv_kb_per_token": 64,
     "max_ctx": 262144, "mtp": True, "vision": True,
     "blurb": "Alibaba's dense 27B with a built-in MTP draft head. Best all-rounder for two 16 GB cards."},
    {"id": "gpt-oss-20b", "name": "gpt-oss-20b", "repo": "unsloth/gpt-oss-20b-GGUF",
     "params_b": 21, "arch": "MoE (3.6B active)", "kv_kb_per_token": 24,
     "max_ctx": 131072, "mtp": False, "vision": False,
     "blurb": "OpenAI's open-weight MoE. Very fast on a single 16 GB card; ships MXFP4."},
    {"id": "gpt-oss-120b", "name": "gpt-oss-120b", "repo": "unsloth/gpt-oss-120b-GGUF",
     "params_b": 117, "arch": "MoE (5.1B active)", "kv_kb_per_token": 36,
     "max_ctx": 131072, "mtp": False, "vision": False,
     "blurb": "Big MoE; ~60 GB weights. Needs CPU offload of experts on this box (slow) - included for reference."},
    {"id": "gemma-3-27b", "name": "Gemma 3 27B IT", "repo": "unsloth/gemma-3-27b-it-GGUF",
     "params_b": 27, "arch": "dense", "kv_kb_per_token": 160,
     "max_ctx": 131072, "mtp": False, "vision": True,
     "blurb": "Google's 27B instruct model with vision. Heavier KV cache than Qwen3.8."},
    {"id": "qwen3.6-35b-a3b", "name": "Qwen3.6-35B-A3B", "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
     "params_b": 35, "arch": "MoE hybrid (3B active; Gated DeltaNet + gated attention, 256 experts)", "kv_kb_per_token": 20,
     "max_ctx": 262144, "mtp": False, "vision": True, "draft_repo": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
     "blurb": "Fast hybrid MoE with a tiny KV cache. UD-Q4_K_XL ~22 GB fits on two 16 GB cards; MTP draft head ships as a separate GGUF."},
    {"id": "qwen3.8-flash-next", "name": "Qwen3.8-Flash-Next", "repo": "unsloth/Qwen3.8-Flash-Next-GGUF",
     "params_b": 177, "arch": "MoE hybrid (125B LM + 51B n-gram embedding, 6B active; Gated DeltaNet + sparse attention, 512 experts)", "kv_kb_per_token": 24,
     "max_ctx": 262144, "mtp": False, "vision": True, "draft_repo": "unsloth/Qwen3.8-Flash-Next-GGUF",
     "blurb": "Experimental 125B/6B-active MoE. UD-Q4_K_XL is 111 GB: needs experts in system RAM (--n-cpu-moe) on any consumer GPU box; MTP draft is a separate GGUF in the same repo."},
    {"id": "qwen3-coder-30b-a3b", "name": "Qwen3-Coder-30B-A3B", "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
     "params_b": 30, "arch": "MoE (3.3B active)", "kv_kb_per_token": 96,
     "max_ctx": 262144, "mtp": False, "vision": False,
     "blurb": "Coding-focused MoE; extremely fast decode, weights ~18 GB at Q4."},
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def run(cmd, timeout=20, shell=False):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=shell,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:  # noqa
        return -1, "", str(e)


def which(name):
    return shutil.which(name)


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)
    except Exception:
        pass


def gb(x):
    return round(x / (1024 ** 3), 2)


# ---------------------------------------------------------------------------
# hardware detection
# ---------------------------------------------------------------------------
def detect_cpu():
    name, cores, threads = platform.processor() or "", None, os.cpu_count()
    if IS_WIN:
        rc, out, _ = run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Select-Object -First 1 | "
                          "ForEach-Object { $_.Name + '|' + $_.NumberOfCores + '|' + $_.NumberOfLogicalProcessors })"])
        if rc == 0 and "|" in out:
            n, c, t = out.split("|")[:3]
            name, cores, threads = n.strip(), int(c), int(t)
    else:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        rc, out, _ = run(["lscpu", "-p=Core,Socket"])
        if rc == 0:
            cores = len({l for l in out.splitlines() if l and not l.startswith("#")})
    flags = ""
    if not IS_WIN:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags"):
                        flags = line
                        break
        except Exception:
            pass
    return {"name": name, "cores": cores or threads, "threads": threads,
            "avx512": ("avx512f" in flags) if flags else None}


def detect_ram():
    if IS_WIN:
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return {"total_gb": gb(m.ullTotalPhys), "free_gb": gb(m.ullAvailPhys)}
    tot = free = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    tot = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable"):
                    free = int(line.split()[1]) * 1024
    except Exception:
        pass
    return {"total_gb": gb(tot), "free_gb": gb(free)}


# Win32_VideoController.AdapterRAM is a uint32 (caps at 4 GB - a 24 GB Arc Pro B60 shows as 4 GB), so the
# real frame-buffer size comes from HardwareInformation.qwMemorySize in the display-class registry keys,
# matched to the WMI adapters by DriverDesc. Output: Name|bytes|DriverVersion per adapter.
WIN_GPU_QUERY = (
    "$c='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}'; $r=@{}; "
    "Get-ChildItem $c -ErrorAction SilentlyContinue | ForEach-Object { "
    "$p=Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue; if($p -and $p.DriverDesc){ "
    "$m=$p.'HardwareInformation.qwMemorySize'; if(-not $m){$m=$p.'HardwareInformation.MemorySize'}; "
    "if($m -is [byte[]]){ $m = if($m.Length -ge 8){[BitConverter]::ToInt64($m,0)}else{[BitConverter]::ToUInt32($m,0)} }; "
    "if($m -and -not $r.ContainsKey($p.DriverDesc)){ $r[$p.DriverDesc]=[int64]$m } } }; "
    "Get-CimInstance Win32_VideoController | ForEach-Object { $n=$_.Name.Trim(); "
    "$v= if($r.ContainsKey($n)){$r[$n]}else{[int64]$_.AdapterRAM}; $n + '|' + $v + '|' + $_.DriverVersion }"
)


def is_integrated_gpu(name, vendor, vram_gb):
    """Integrated graphics ("Intel(R) Graphics" on Core Ultra, UHD / Iris Xe, "AMD Radeon(TM) Graphics") are
    display adapters, not compute targets. Discrete Arc / Radeon cards with a real frame buffer are."""
    if vendor == "nvidia":
        return False
    if re.search(r"ASPEED|Matrox|BMC", name, re.I):      # server-board BMC VGA (ASRock Rack ROMED8-2T etc.)
        return True
    if re.match(r"Intel\(R\) (Graphics|UHD|Iris|HD)", name) or re.search(r"Radeon\(TM\) Graphics$", name):
        return True
    if re.search(r"Arc|Radeon (RX|Pro)|Instinct", name, re.I):
        return vram_gb < 3
    return vram_gb < 3


def linux_gpu_vram_gb(lspci_row):
    """Best-effort VRAM for a non-NVIDIA card on Linux: amdgpu exposes mem_info_vram_total; the Intel xe/i915
    drivers expose the frame buffer as the largest 64-bit prefetchable BAR (resizable BAR = full VRAM)."""
    try:
        bdf = lspci_row.split()[0]
        if bdf.count(":") == 1:
            bdf = "0000:" + bdf
        dev = f"/sys/bus/pci/devices/{bdf}"
        p = f"{dev}/mem_info_vram_total"
        if os.path.exists(p):
            with open(p) as f:
                return round(int(f.read().strip()) / 1024 ** 3, 1)
        best = 0
        with open(f"{dev}/resource") as f:
            for line in f:
                a, b, _ = (int(x, 16) for x in line.split())
                if b > a:
                    best = max(best, b - a + 1)
        return round(best / 1024 ** 3, 1) if best >= 1024 ** 3 else 0
    except Exception:
        return 0


def detect_gpus():
    gpus = []
    smi = which("nvidia-smi") or (r"C:\Windows\System32\nvidia-smi.exe" if IS_WIN else None)
    if smi and os.path.exists(smi):
        rc, out, _ = run([smi, "--query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version,pcie.link.width.current",
                          "--format=csv,noheader,nounits"])
        if rc == 0:
            for row in out.splitlines():
                f = [x.strip() for x in row.split(",")]
                if len(f) >= 6 and f[0].isdigit():
                    gpus.append({"vendor": "nvidia", "index": int(f[0]), "name": f[1],
                                 "vram_gb": round(int(f[2]) / 1024, 1), "used_gb": round(int(f[3]) / 1024, 1),
                                 "compute_cap": f[4], "driver": f[5], "pcie_width": f[6] if len(f) > 6 else "",
                                 "integrated": False})
    # non-NVIDIA (Intel Arc / AMD discrete cards + integrated GPUs)
    if IS_WIN:
        rc, out, _ = run(["powershell", "-NoProfile", "-Command", WIN_GPU_QUERY])
        if rc == 0:
            for row in out.splitlines():
                p = row.split("|")
                if len(p) < 3 or re.search(r"NVIDIA|Microsoft Basic|Remote|Virtual|DisplayLink", p[0], re.I):
                    continue
                name = p[0].strip()
                vendor = "intel" if "Intel" in name else "amd" if re.search(r"AMD|Radeon|ATI", name) else "other"
                try:
                    vram = round(int(p[1]) / 1024 ** 3, 1)
                except Exception:
                    vram = 0
                gpus.append({"vendor": vendor, "index": len(gpus), "name": name, "vram_gb": vram,
                             "used_gb": None, "compute_cap": "", "driver": p[2].strip(), "pcie_width": "",
                             "integrated": is_integrated_gpu(name, vendor, vram)})
    else:
        rc, out, _ = run(["lspci"])
        if rc == 0:
            for row in out.splitlines():
                if re.search(r"VGA|3D|Display", row) and "NVIDIA" not in row:
                    name = row.split(":", 2)[-1].strip()
                    vendor = "intel" if "Intel" in name else "amd" if re.search(r"AMD|ATI|Radeon", name) else "other"
                    vram = linux_gpu_vram_gb(row)
                    gpus.append({"vendor": vendor, "index": len(gpus), "name": name, "vram_gb": vram,
                                 "used_gb": None, "compute_cap": "", "driver": "", "pcie_width": "",
                                 "integrated": is_integrated_gpu(name, vendor, vram)})
    return gpus


def detect_toolchain():
    tc = {}
    for tool in ["git", "cmake", "ninja", "nvcc", "gcc", "g++", "glslc", "nvidia-smi", "python"]:
        p = which(tool)
        ver = ""
        if p:
            flag = "--version" if tool not in ("nvidia-smi",) else "--version"
            rc, out, err = run([p, flag], timeout=10)
            first = (out or err).splitlines()[0] if (out or err) else ""
            m = re.search(r"(\d+\.\d+(\.\d+)?)", first)
            ver = m.group(1) if m else first[:40]
        tc[tool] = {"path": p, "version": ver}
    if IS_WIN:
        vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        vs = ""
        if os.path.exists(vswhere):
            rc, out, _ = run([vswhere, "-latest", "-products", "*", "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
            vs = out.strip()
        tc["msvc"] = {"path": vs, "version": "installed" if vs else ""}
        cuda_root = os.environ.get("CUDA_PATH", "")
        if not tc["nvcc"]["path"] and cuda_root and os.path.exists(os.path.join(cuda_root, "bin", "nvcc.exe")):
            tc["nvcc"] = {"path": os.path.join(cuda_root, "bin", "nvcc.exe"), "version": ""}
        tc["vulkan_sdk"] = {"path": os.environ.get("VULKAN_SDK", ""), "version": ""}
    oa = oneapi_root()
    ver = ""
    if oa:
        try:   # <root>/compiler/<version>/ - "latest" is a symlink/junction to the newest
            ver = max((d for d in os.listdir(os.path.join(oa, "compiler")) if d[:1].isdigit()), default="")
        except Exception:
            pass
    tc["oneapi"] = {"path": oa, "version": ver,
                    "ok_for_arc_b": bool(ver) and tuple(int(x) for x in ver.split(".")[:2]) >= (2025, 2)}   # bmg_g31 enum
    return tc


def build_info():
    p = os.path.join(BIN_DIR, "build-info.json")
    try:
        with open(p, encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        info = None
    server = os.path.join(BIN_DIR, "llama-server" + EXE)
    return {"present": os.path.exists(server), "server_path": server, "info": info,
            "mtime": os.path.getmtime(server) if os.path.exists(server) else None}


def compute_gpus(gpus, backend):
    """GPUs llama.cpp will actually use for a backend: CUDA -> NVIDIA only; Vulkan -> discrete cards
    (ggml-vulkan itself defaults to discrete GPUs only, so integrated graphics are display adapters here);
    CPU -> none."""
    if backend == "cuda":
        return [g for g in gpus if g["vendor"] == "nvidia"]
    if backend == "vulkan":
        return [g for g in gpus if not g.get("integrated") and g["vram_gb"] >= 3]
    if backend == "sycl":
        return [g for g in gpus if g["vendor"] == "intel" and not g.get("integrated") and g["vram_gb"] >= 3]
    return []


def oneapi_root():
    """Intel oneAPI install (needed to build and to run the SYCL backend), or ''."""
    cands = [os.environ.get("ONEAPI_ROOT", "")]
    if IS_WIN:
        cands += [r"C:\Program Files (x86)\Intel\oneAPI", r"C:\Program Files\Intel\oneAPI"]
    else:
        cands += ["/opt/intel/oneapi"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "setvars.bat" if IS_WIN else "setvars.sh")):
            return c
    return ""


_oneapi_env_cache = {}


def oneapi_env(root=""):
    """Environment exported by oneAPI's setvars script (PATH with the SYCL / oneMKL / oneDNN runtime DLLs
    or .so files, CMPLR_ROOT, ...). SYCL-built llama-server will not start without it."""
    root = root or oneapi_root()
    if not root:
        return {}
    if root in _oneapi_env_cache:
        return _oneapi_env_cache[root]
    if IS_WIN:
        # A temp .cmd avoids the cmd.exe quoting trap: subprocess escapes embedded quotes as \" which cmd does
        # not understand, so the direct 'cmd /s /c "..."' form silently ran nothing and no DLL path was imported.
        wrapper = os.path.join(tempfile.gettempdir(), "aios-setvars.cmd")
        with open(wrapper, "w", encoding="ascii") as f:
            f.write(f'@echo off\r\ncall "{os.path.join(root, "setvars.bat")}" intel64 >nul 2>&1\r\nset\r\n')
        cmd = ["cmd.exe", "/c", wrapper]
    else:
        cmd = ["bash", "-c", f'source "{os.path.join(root, "setvars.sh")}" >/dev/null 2>&1; env']
    rc, out, _ = run(cmd, timeout=180)
    env = {}
    if rc == 0:
        for line in out.splitlines():
            k, sep, v = line.partition("=")
            if sep and k and not k.startswith("="):
                env[k] = v
    # setvars sets CMPLR_ROOT / ONEAPI_ROOT; if they are missing the import failed and the result must not be cached
    if not any(k.upper() in ("CMPLR_ROOT", "ONEAPI_ROOT") for k in env):
        return {}
    _oneapi_env_cache[root] = env
    return env


_hw_cache = {"t": 0, "gpus": None, "cpu": None, "tc": None}


def parse_gpu_selection(sel):
    """'0,2' -> {0, 2}; '' / None / 'all' -> None (use every compute GPU)."""
    if not sel or str(sel).strip().lower() in ("all", "*"):
        return None
    try:
        return {int(x) for x in re.split(r"[,\s;]+", str(sel).strip()) if x != ""}
    except ValueError:
        return None


def detect_all(backend=None, fresh=False, gpu_selection_arg=None):
    """Hardware + backend view. `gpus` = optional selection of GPU indexes (as shown in the GUI) to serve on;
    the rest of the cards stay free for another llama-server instance (one model per card, llama-swap style)."""
    if fresh or time.time() - _hw_cache["t"] > 30:
        _hw_cache.update(t=time.time(), gpus=detect_gpus(), cpu=detect_cpu(), tc=detect_toolchain())
    gpus = [dict(g) for g in _hw_cache["gpus"]]
    nv = [g for g in gpus if g["vendor"] == "nvidia"]
    # cuda for NVIDIA; sycl (Intel's native backend) when a discrete Arc card and oneAPI are both present;
    # otherwise vulkan for any discrete GPU (works on Arc / AMD with no extra toolkit); else cpu
    if nv:
        rec = "cuda"
    elif compute_gpus(gpus, "sycl") and _hw_cache["tc"].get("oneapi", {}).get("path"):
        rec = "sycl"
    else:
        rec = "vulkan" if compute_gpus(gpus, "vulkan") else "cpu"
    backend = backend if backend in ("cuda", "vulkan", "sycl", "cpu") else rec
    all_use = compute_gpus(gpus, backend)
    sel = parse_gpu_selection(gpu_selection_arg)
    use = [g for g in all_use if sel is None or g["index"] in sel] or all_use
    for g in gpus:
        g["compute"] = g in use
    # device ordinal inside the backend (CUDA_VISIBLE_DEVICES / GGML_VK_VISIBLE_DEVICES / ONEAPI_DEVICE_SELECTOR
    # count only the devices that backend exposes, in the order llama.cpp lists them - assumed to match the
    # WMI/lspci order, which holds for identical cards)
    device_ids = [all_use.index(g) for g in use]
    return {
        "all_compute_gpus": all_use, "device_ids": device_ids,
        "gpu_selection": sorted(sel) if sel is not None and len(use) < len(all_use) else None,
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version(),
               "machine": platform.machine(), "python": platform.python_version(), "hostname": socket.gethostname()},
        "cpu": _hw_cache["cpu"], "ram": detect_ram(), "gpus": gpus, "compute_gpus": use,
        "vram_total_gb": round(sum(g["vram_gb"] for g in use), 1),
        "vram_min_gb": min([g["vram_gb"] for g in use] or [0]),
        "recommended_backend": rec, "backend": backend,
        "toolchain": _hw_cache["tc"], "build": build_info(), "root": ROOT, "version": APP_VERSION,
        "scripts": script_info(),
    }


def script_info():
    """Which build script AIOS will run, and whether it is the current version (supports -Backend)."""
    name = "build-llama.ps1" if IS_WIN else "build-llama.sh"
    path = os.path.join(ROOT, name)
    v = script_version(path)
    return {"name": name, "path": path, "present": v >= 0, "version": v, "expected": SCRIPTS_VERSION,
            "supports_backend": v >= SCRIPTS_VERSION}


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------
_hf_cache = {}


def hf_files(repo):
    """List GGUF files in a repo (with sizes) via the HF API."""
    if repo in _hf_cache and time.time() - _hf_cache[repo][0] < 600:
        return _hf_cache[repo][1]
    url = f"{HF_BASE}/api/models/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": HF_UA})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=30) as r:
        items = json.load(r)
    files = []
    for it in items:
        path = it.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        size = it.get("size") or (it.get("lfs") or {}).get("size") or 0
        files.append({"path": path, "size": size, "size_gb": gb(size)})
    # group into quants: mmproj separate; shards grouped by quant dir/prefix
    quants = {}
    mmproj = [f for f in files if "mmproj" in f["path"].lower()]
    drafts = [f for f in files if re.search(r"mtp|draft", os.path.basename(f["path"]), re.I) and "mmproj" not in f["path"].lower()]
    extras = [f for f in files if re.search(r"imatrix", f["path"], re.I)] + drafts
    for f in files:
        lp = f["path"].lower()
        if "mmproj" in lp or f in extras:
            continue
        base = os.path.basename(f["path"])
        m = re.search(r"-((?:UD-)?(?:IQ|Q|MXFP|BF|F)[0-9A-Za-z_]+?)(?:-\d{5}-of-\d{5})?\.gguf$", base)
        tag = m.group(1) if m else base.replace(".gguf", "")
        q = quants.setdefault(tag, {"tag": tag, "files": [], "size": 0})
        q["files"].append(f["path"])
        q["size"] += f["size"]
    out = []
    for q in quants.values():
        q["size_gb"] = gb(q["size"])
        q["sharded"] = len(q["files"]) > 1
        q["files"].sort()
        out.append(q)
    out.sort(key=lambda q: q["size"], reverse=True)
    res = {"repo": repo, "quants": out, "mmproj": mmproj, "drafts": drafts, "extras": [e["path"] for e in extras]}
    _hf_cache[repo] = (time.time(), res)
    return res


def quant_bits(tag):
    t = tag.upper()
    if t in ("F16", "BF16", "F32"):
        return 16
    m = re.search(r"(?:^|-)(?:I?Q|MXFP)(\d)", t)
    return int(m.group(1)) if m else 0


def quant_penalty(tag):
    """Lower is preferred among equal bit-widths: K-quants/UD first, legacy Qx_0/Qx_1 and IQ last."""
    t = tag.upper()
    if re.search(r"Q\d_[01]$", t):
        return 2
    if t.startswith("UD-IQ") or t.startswith("IQ"):
        return 1
    return 0


_rel_cache = {"t": 0, "data": None}


def llama_releases(limit=40):
    """Recent llama.cpp release tags from GitHub (cached 10 min). Empty list if GitHub is unreachable."""
    if _rel_cache["data"] is not None and time.time() - _rel_cache["t"] < 600:
        return _rel_cache["data"]
    url = f"{GH_API}/repos/{LLAMA_REPO}/releases?per_page={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": HF_UA, "Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    out = []
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            for rel in json.load(r):
                out.append({"tag": rel.get("tag_name"), "name": rel.get("name") or rel.get("tag_name"),
                            "date": (rel.get("published_at") or "")[:10], "prerelease": bool(rel.get("prerelease")),
                            "url": rel.get("html_url")})
        _rel_cache.update(t=time.time(), data=out)
    except Exception as e:  # noqa
        out = {"error": str(e)}
    return out


def recommend(hw, entry, quants, ctx=65536, kv="q8_0", priority="speed"):
    """Pick a quant that fits detected VRAM (all GPUs, layer split) leaving room for KV cache + buffers.
    priority=speed  -> the best 4-bit quant that fits (decode is bandwidth-bound: fewer bytes = more tok/s)
    priority=quality-> the largest quant that fits"""
    vram = hw["vram_total_gb"]
    ngpu = max(1, len(hw.get("compute_gpus", hw["gpus"])))
    kv_factor = {"f16": 1.0, "q8_0": 0.5, "q4_0": 0.25}.get(kv, 0.5)
    kv_gb = entry["kv_kb_per_token"] * ctx * kv_factor / (1024 ** 2)
    overhead = 1.2 * ngpu + (1.0 if entry.get("mtp") else 0)   # compute buffers, CUDA graphs, draft head
    budget = vram * (0.90 if hw.get("backend") == "sycl" else 0.94) - kv_gb - overhead
    fits = [q for q in quants if q["size_gb"] <= budget]
    moe = "moe" in (entry.get("arch") or "").lower()
    # MoE: experts can live in RAM (--n-cpu-moe) while attention + shared weights stay on the cards
    offload_budget = budget + hw["ram"]["total_gb"] * 0.6 if moe else budget
    offload = [q["tag"] for q in quants if budget < q["size_gb"] <= offload_budget]
    by_quality = sorted(fits, key=lambda q: (quant_bits(q["tag"]) >= 16, quant_penalty(q["tag"]), -q["size"]))
    best = None
    if fits:
        if priority == "quality":
            best = by_quality[0]
        else:
            four = [q for q in fits if quant_bits(q["tag"]) == 4]
            pool = four or [q for q in fits if 4 < quant_bits(q["tag"]) < 16] or fits
            best = sorted(pool, key=lambda q: (quant_penalty(q["tag"]), -q["size"] if four else q["size"]))[0]
    ram_budget = hw["ram"]["total_gb"] * 0.7 + vram * 0.9 - kv_gb
    cpu_fallback = None if best else next((q for q in sorted(quants, key=lambda q: -q["size"]) if q["size_gb"] <= ram_budget), None)
    return {"best": best["tag"] if best else None, "cpu_fallback": cpu_fallback["tag"] if cpu_fallback else None,
            "budget_gb": round(budget, 1), "kv_gb": round(kv_gb, 2), "overhead_gb": round(overhead, 1),
            "priority": priority, "fits": [q["tag"] for q in by_quality], "offload": offload}


# bytes per weight for the quant tiers the catalog recommends from (GGUF typical, incl. overhead)
QUANT_TIERS = [("Q8_0", 1.07), ("Q6_K", 0.83), ("Q5_K_M", 0.72), ("UD-Q4_K_XL", 0.63), ("Q4_K_M", 0.60), ("IQ4_XS", 0.54),
               ("UD-Q3_K_XL", 0.50), ("IQ3_XXS", 0.41), ("UD-Q2_K_XL", 0.36)]


def recommend_models(hw, ctx=32768, kv="q8_0", priority="speed"):
    """Rank the catalog for THIS machine: which quant tier fits, on how many of the detected cards, or only with MoE
    experts in RAM. Sizes are estimated from params_b (the quant table uses real file sizes once a repo is opened)."""
    use = hw.get("compute_gpus") or []
    ncards = len(use)
    per_card = min([g["vram_gb"] for g in use] or [0])
    vram = hw.get("vram_total_gb") or 0
    ram = (hw.get("ram") or {}).get("total_gb") or 0
    be = hw.get("backend")
    factor = 0.90 if be == "sycl" else 0.94
    kv_factor = {"f16": 1.0, "q8_0": 0.5, "q4_0": 0.25}.get(kv, 0.5)
    out = []
    for c in CATALOG:
        moe = "moe" in (c.get("arch") or "").lower()
        kv_gb = c["kv_kb_per_token"] * ctx * kv_factor / (1024 ** 2)
        verdict = None
        for tag, bpw in QUANT_TIERS:
            size = c["params_b"] * bpw
            if priority == "speed" and tag in ("Q8_0", "Q6_K", "Q5_K_M") and c["params_b"] * 0.63 + kv_gb + 1.2 <= vram * factor:
                continue   # speed priority: prefer the 4-bit tier when it fits
            # fewest cards that hold it (layer split): buffers 1.2 GB per card
            for n in range(1, ncards + 1):
                if size + kv_gb + 1.2 * n <= per_card * n * factor:
                    verdict = {"tag": tag, "size_gb": round(size, 1), "cards": n, "mode": "single card" if n == 1 else f"layer split over {n} of {ncards}",
                               "fit": "fits", "rank": 0 if n == 1 else 1}
                    break
            if verdict:
                break
        if moe and ncards and (not verdict or quant_bits(verdict["tag"]) < 4):
            # MoE: a 4-bit tier with experts in RAM beats a 2/3-bit tier squeezed onto the cards
            gpu_verdict = verdict
            verdict = None
            for tag, bpw in QUANT_TIERS:
                if priority == "speed" and quant_bits(tag) > 4:
                    continue   # experts stream from RAM: smaller tiers are faster, 4-bit is the quality floor worth paying for
                size = c["params_b"] * bpw
                in_ram = size + kv_gb + 1.2 * ncards - vram * factor
                if in_ram <= ram * 0.6:
                    verdict = {"tag": tag, "size_gb": round(size, 1), "cards": ncards, "mode": f"experts in RAM (~{max(1, int(in_ram))} GB, --n-cpu-moe)",
                               "fit": "offload", "rank": 2}
                    break
            if gpu_verdict and (not verdict or quant_bits(verdict["tag"]) <= quant_bits(gpu_verdict["tag"])):
                verdict = gpu_verdict
        if not verdict:
            size = c["params_b"] * QUANT_TIERS[-1][1]
            verdict = {"tag": None, "size_gb": round(size, 1), "cards": 0, "mode": "does not fit this machine", "fit": "none", "rank": 3}
        # order: fits > experts-in-RAM > too big; speed priority keeps models that leave cards free ahead of ones that
        # need (nearly) every card; then the most capable model first, MoE ahead of dense at equal size for speed
        frac = verdict["cards"] / ncards if ncards else 1
        verdict["sort"] = ({"fits": 0, "offload": 1, "none": 2}[verdict["fit"]],
                           1 if (priority == "speed" and verdict["fit"] == "fits" and frac > 0.6 and ncards >= 3) else 0,
                           -c["params_b"], 0 if moe else 1)
        out.append(dict(c, verdict=verdict, moe=moe))
    out.sort(key=lambda c: c["verdict"]["sort"])
    if out and out[0]["verdict"]["fit"] == "fits":
        out[0]["verdict"]["best"] = True
    return out


def model_size_gb(model_path):
    """Size of a GGUF including its other shards (model-00001-of-00004.gguf -> all four)."""
    try:
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", model_path, re.I)
        if m:
            paths = [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, int(m.group(3)) + 1)]
            return sum(os.path.getsize(p) for p in paths if os.path.exists(p)) / 1024 ** 3
        return os.path.getsize(model_path) / 1024 ** 3
    except Exception:
        return 0.0


def server_env(hw, entry, opts):
    """Environment variables that go with server_args(): device pinning + Arc/SYCL runtime knobs."""
    env = {}
    be = hw.get("backend")
    ids = hw.get("device_ids") or []
    if hw.get("gpu_selection") is not None and ids:
        if be == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
        elif be == "vulkan":
            env["GGML_VK_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
        elif be == "sycl":
            env["ONEAPI_DEVICE_SELECTOR"] = ";".join(f"level_zero:{i}" for i in ids)
    if be == "vulkan" and any(g["vendor"] == "amd" and g.get("integrated") for g in hw.get("gpus", [])):
        # Ryzen iGPU ("AMD Radeon(TM) Graphics"): keep the RADV/AMD ICD out of the Vulkan loader so only the
        # discrete cards are enumerated (the iGPU is already excluded from compute, this stops the driver load)
        env["VK_LOADER_DRIVERS_DISABLE"] = "*radeon*"
    if be == "sycl":
        env["ZES_ENABLE_SYSMAN"] = "1"                        # free-memory queries across several cards
        env["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"   # single allocations > 4 GiB (big KV cache / tensors)
        if "moe" in (entry.get("arch") or "").lower():
            env["GGML_SYCL_DISABLE_OPT"] = "1"                # Warren's Arc benchmarks: MoE is faster without the opt path
    return env


def server_args(hw, entry, model_path, opts):
    """Build the tuned llama-server argument list from hardware + model + user overrides."""
    use = hw.get("compute_gpus", hw["gpus"])
    ngpu = len(use)
    be = hw.get("backend")
    arc = be in ("vulkan", "sycl") and any(g["vendor"] == "intel" for g in use)
    moe = "moe" in (entry.get("arch") or "").lower()
    cores = hw["cpu"].get("cores") or 8
    ctx = int(opts.get("ctx") or 65536)
    kv = opts.get("kv") or "q8_0"
    fa = opts.get("fa") or ("off" if (be == "sycl" and moe) else "on")   # SYCL + MoE: flash-attn path is slower / unstable on Arc
    if fa not in ("on", "off", "auto"):
        fa = "on"
    ub = str(opts.get("ub") or ("1024" if moe else "2048" if arc else "512"))   # Arc: big -ub = +60% prefill; MoE: 1024 keeps expert gathers small
    ncmoe = int(opts.get("ncmoe") or 0)
    a = ["-m", model_path, "--jinja", "-c", str(ctx), "-fa", fa,
         "--cache-type-k", kv, "--cache-type-v", kv if fa == "on" else "f16",   # a quantized V cache needs flash-attn
         "-b", "2048", "-ub", ub,
         "-t", str(cores),
         "--host", opts.get("host") or "0.0.0.0", "--port", str(opts.get("port") or 8080), "--metrics"]
    if moe:
        a += ["--cache-reuse", "256", "--defrag-thold", "0.1"]   # MoE serving: reuse prompt prefixes across slots, keep KV compact
    if ncmoe > 0:
        a += ["--n-cpu-moe", str(ncmoe)]                         # experts of the first N layers stay in system RAM
    # Load mode (llama.cpp replaced --no-mmap with -lm/--load-mode: auto|none|mmap|mlock|mmap+mlock|dio).
    # On the Arc boxes reading the weights in (none) avoids RAM spill-over once they are on the cards, but only
    # when they fit in RAM (4x B60 = 96 GB VRAM vs 48 GB RAM: a 60 GB model must stream via mmap = auto).
    msize = model_size_gb(model_path)
    lm = opts.get("load_mode") or ("none" if arc and ncmoe == 0 and msize > 0 and msize + 6 < (hw.get("ram", {}).get("total_gb") or 0) else "auto")
    if lm != "auto":
        a += ["--load-mode", lm]
    ngl = opts.get("ngl", "999")
    a += ["-ngl", str(ngl)]
    if ngpu > 1:
        split = opts.get("split") or "layer"
        if be == "sycl" and split == "row":
            split = "layer"                                   # row split is a CUDA thing; SYCL offers layer / tensor(2 cards)
        a += ["-sm", split, "-mg", "0"]
    elif be == "sycl":
        # SYCL sees the iGPU as a device too; docs/backend/SYCL.md: single card = "-sm none -mg 0"
        # (the discrete GPU is listed first)
        a += ["-sm", "none", "-mg", "0"]
    par = int(opts.get("parallel") or 1)
    a += ["--parallel", str(par)]
    draft = opts.get("draft") or ""
    spec = opts.get("spec") or ("mtp" if entry.get("mtp") else "draft" if draft else "none")
    if spec == "mtp" and entry.get("mtp") and opts.get("mtp", True) and par == 1:
        a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    elif spec == "draft" and draft and os.path.exists(draft):
        # separate MTP/draft GGUF (-md). spec_kind: 'draft-mtp' for an MTP head file, plain 'draft' for a small draft model
        kind = opts.get("spec_kind") or ("draft-mtp" if re.search(r"mtp", os.path.basename(draft), re.I) else "draft")
        a += ["-md", draft, "-ngld", "99", "--spec-type", kind, "--spec-draft-n-max", "4" if kind == "draft-mtp" else "8"]
    elif spec == "ngram":
        a += ["--spec-type", "ngram-simple", "--spec-draft-n-max", "16"]   # draft-model-free lookup decoding: helps repetitive code/agent output
    if opts.get("mmproj"):
        a += ["--mmproj", opts["mmproj"]]
    if opts.get("extra"):
        extra = re.sub(r"(?<!\S)--?\s+(?=[A-Za-z])", lambda m: m.group(0).strip(), opts["extra"])  # "-- temp" -> "--temp"
        a += extra.split()
    return a


# ---------------------------------------------------------------------------
# jobs (build / download) with live log
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, kind, title):
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.title = kind, title
        self.lines, self.status, self.started, self.ended = [], "running", time.time(), None
        self.progress = None  # 0..1 for downloads
        self.result = None
        self.proc = None
        self.lock = threading.Lock()

    def log(self, s):
        with self.lock:
            self.lines.append(s.rstrip("\n"))

    def finish(self, ok, msg=""):
        if self.status != "running":   # already finished (e.g. cancelled) - keep the first verdict
            return
        self.status = "done" if ok else "failed"
        self.ended = time.time()
        if msg:
            self.log(msg)

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "started": self.started, "ended": self.ended, "progress": self.progress, "lines": len(self.lines),
                "result": self.result}


JOBS = {}


def start_job(kind, title, target):
    j = Job(kind, title)
    JOBS[j.id] = j

    def wrap():
        try:
            target(j)
        except Exception as e:  # noqa
            j.finish(False, f"!! {e}")

    threading.Thread(target=wrap, daemon=True).start()
    return j


def kill_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    if IS_WIN:
        run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], timeout=15)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:
            proc.kill()


def stream_proc(job, cmd, env=None, cwd=None):
    job.log("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=env, cwd=cwd, encoding="utf-8", errors="replace",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                         start_new_session=not IS_WIN)
    job.proc = p
    for line in p.stdout:
        job.log(re.sub(r"\x1b\[[0-9;]*m", "", line))
    p.wait()
    return p.returncode


def job_build(job, backend, clean, branch):
    env = dict(os.environ, LLAMA_BACKEND=backend or "auto")
    if IS_WIN:
        script = os.path.join(ROOT, "build-llama.ps1")
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Backend", backend or "auto",
               "-SrcDir", os.path.join(ROOT, "llama.cpp"), "-InstallDir", BIN_DIR]
        if clean:
            cmd.append("-Clean")
        if branch:
            cmd += ["-Branch", branch]
    else:
        script = os.path.join(ROOT, "build-llama.sh")
        cmd = ["bash", script, "--backend", backend or "auto"]
        if clean:
            cmd.append("--clean")
        if branch:
            cmd += ["--branch", branch]
    ensure_scripts(job.log)
    si = script_info()
    if not si["supports_backend"]:
        job.finish(False, f"!! could not write a current {si['name']} into {ROOT} (check folder permissions)")
        return
    rc = stream_proc(job, cmd, env=env, cwd=ROOT)
    job.finish(rc == 0, "== build finished OK" if rc == 0 else f"!! build failed (exit {rc})")


DL_CONNECTIONS = max(1, min(16, int(os.environ.get("AIOS_DL_CONNECTIONS", "8"))))


def _hf_request(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": HF_UA})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    return req


def _download_file(job, url, dest, size_hint, progress_cb):
    """Multi-connection ranged download with resume. Hugging Face's CDN caps a single stream at roughly
    20-60 MB/s; 8 parallel ranges normally saturate a gigabit link. State lives in <dest>.part + <dest>.parts.json."""
    part, meta = dest + ".part", dest + ".parts.json"
    # HEAD-ish probe: size + range support
    r = urllib.request.urlopen(_hf_request(url, (0, 0)), timeout=60)
    ranged = r.status == 206
    cr = r.headers.get("Content-Range", "")
    size = int(cr.rsplit("/", 1)[-1]) if ranged and "/" in cr else int(r.headers.get("Content-Length") or size_hint or 0)
    r.close()
    if not ranged or size <= 64 * 1024 * 1024 or DL_CONNECTIONS == 1:
        # small file / no range support: single stream
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = _hf_request(url, (have, size - 1) if have and size else None)
        r = urllib.request.urlopen(req, timeout=60)
        mode = "ab" if have and r.status == 206 else "wb"
        if mode == "wb":
            have = 0
        with open(part, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk); have += len(chunk); progress_cb(have)
        os.replace(part, dest)
        return have
    # segments
    n = DL_CONNECTIONS
    seg = -(-size // n)
    segments = [[i * seg, min(size, (i + 1) * seg) - 1, 0] for i in range(n)]   # [start, end, done]
    if os.path.exists(part) and os.path.exists(meta):
        try:
            with open(meta) as f:
                saved = json.load(f)
            if saved.get("size") == size and len(saved.get("segments", [])) == n:
                segments = saved["segments"]
                job.log(f"== resuming {os.path.basename(dest)} at {gb(sum(x[2] for x in segments))} GB")
        except Exception:
            pass
    with open(part, "ab") as f:
        f.truncate(size)
    lock = threading.Lock()
    errors = []

    def save_meta():
        with open(meta, "w") as f:
            json.dump({"size": size, "segments": segments}, f)

    def worker(i):
        st, en, done = segments[i]
        if st + done > en:
            return
        for attempt in range(5):
            try:
                r = urllib.request.urlopen(_hf_request(url, (st + done, en)), timeout=60)
                with open(part, "r+b") as f:
                    f.seek(st + done)
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        with lock:
                            segments[i][2] = done
                return
            except Exception as e:   # noqa
                if job.status != "running":
                    return
                time.sleep(2 * (attempt + 1))
                err = e
        errors.append(f"segment {i}: {err}")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    last_save = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(0.5)
        with lock:
            have = sum(x[2] for x in segments)
        progress_cb(have)
        if time.time() - last_save > 5:
            with lock:
                save_meta()
            last_save = time.time()
        if job.status != "running":
            save_meta()
            raise RuntimeError("cancelled")
    if errors:
        save_meta()
        raise RuntimeError("; ".join(errors))
    have = sum(x[2] for x in segments)
    if have != size:
        save_meta()
        raise RuntimeError(f"incomplete: {have} of {size} bytes")
    os.replace(part, dest)
    try:
        os.remove(meta)
    except OSError:
        pass
    return size


def job_download(job, repo, files, total):
    os.makedirs(MODEL_DIR, exist_ok=True)
    done_bytes = 0
    free = free_disk_gb(MODEL_DIR)
    if free is not None and total and free < total / 1024 ** 3 + 2:
        raise RuntimeError(f"not enough space in {MODEL_DIR}: {free} GB free, need {gb(total)} GB. Point AIOS at another folder (Models folder field).")
    job.log(f"== models folder: {MODEL_DIR}   ({DL_CONNECTIONS} parallel connections; AIOS_DL_CONNECTIONS to change)")
    for path in files:
        dest = os.path.join(MODEL_DIR, os.path.basename(path))
        url = f"{HF_BASE}/{repo}/resolve/main/{urllib.parse.quote(path)}"
        if os.path.exists(dest):
            job.log(f"== already have {os.path.basename(dest)}")
            done_bytes += os.path.getsize(dest)
            job.progress = done_bytes / total if total else None
            continue
        job.log(f"== downloading {os.path.basename(dest)}")
        state = {"last": time.time(), "t0": time.time(), "got0": None}

        def cb(have, state=state, dest=dest):
            if state["got0"] is None:
                state["got0"] = have
            job.progress = (done_bytes + have) / total if total else None
            if time.time() - state["last"] > 2:
                rate = (have - state["got0"]) / max(1e-6, time.time() - state["t0"]) / 1e6
                job.log(f"   {os.path.basename(dest)}: {gb(have)} GB  ({rate:.0f} MB/s)")
                state["last"] = time.time()

        done_bytes += _download_file(job, url, dest, 0, cb)


# ---------------------------------------------------------------------------
# llama-server process management
# ---------------------------------------------------------------------------
EXIT_HINTS = {
    0xC0000135: "STATUS_DLL_NOT_FOUND: a runtime DLL is missing. SYCL build -> the oneAPI environment (setvars.bat) was not "
                "loaded; CUDA build -> cudart64/cublas DLLs; any build -> Visual C++ 2022 redistributable.",
    0xC0000005: "Access violation inside llama-server - most often a driver/backend crash. Try fewer GPU layers, -fa off, "
                "or the other backend, and check the GPU driver version.",
    0xC0000409: "Stack buffer overrun / fail-fast - usually a corrupt GGUF or a quant newer than this llama.cpp build. "
                "Re-download the model or rebuild on master.",
    0xC000013A: "Terminated by Ctrl+C / console close.",
    1: "llama-server rejected an argument or could not load the model - the lines above say which. A stray '--' or a "
       "misspelled flag in Extra flags is the usual cause.",
}



# ---------------------------------------------------------------------------
# Benchmark: settings sweep (llama-bench + slot test) and the "streamer" prompt suite with an LLM judge
# ---------------------------------------------------------------------------
BENCH_FILE = os.path.join(ROOT, "bench-results.json")
BENCH_SYSTEM = ("You are an expert front-end engineer. Answer with ONE complete, self-contained HTML document in a single "
                "```html code block: all CSS and JavaScript inline, no external files except CDN scripts, no explanation "
                "before or after the code block.")
BENCH_PROMPTS = [
    {"id": "browser-os", "name": "Browser OS", "html": True, "max_tokens": 6000, "system": BENCH_SYSTEM,
     "prompt": "Build a desktop operating system UI that runs in the browser: a desktop with icons, a taskbar with a start menu "
               "and a clock, and draggable, resizable windows with minimise/maximise/close. Include at least three working apps: "
               "a notepad that keeps its text while the window is open, a calculator, and a file explorer over an in-memory folder "
               "tree. Multiple windows must be open at once with correct z-ordering.",
     "rubric": "Does it render a desktop with taskbar and clock; can several windows be opened, dragged, resized, brought to front "
               "and closed; do notepad, calculator and file explorer actually work; is the code free of runtime errors?"},
    {"id": "browser-game", "name": "Browser game (Breakout)", "html": True, "max_tokens": 5000, "system": BENCH_SYSTEM,
     "prompt": "Write a complete Breakout game on an HTML5 canvas: paddle controlled by mouse and arrow keys, a ball with "
               "realistic bounce angles depending on where it hits the paddle, 5 rows of coloured bricks, score, 3 lives, "
               "a start screen, game over and win screens with restart, and sound effects generated with the Web Audio API.",
     "rubric": "Is the game playable (paddle moves, ball bounces, bricks break, score/lives update); are start, game-over and win "
               "states handled; does it run without console errors; is the code clean?"},
    {"id": "voxel-world", "name": "Voxel world", "html": True, "max_tokens": 6000, "system": BENCH_SYSTEM,
     "prompt": "Create a Minecraft-style voxel world with Three.js (load it from https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js). "
               "Generate a 32x32 terrain with heights from simple noise, grass/dirt/stone block colours, first-person WASD + mouse-look "
               "controls with pointer lock, jump with gravity and collision against the terrain, and left-click to remove / right-click "
               "to place blocks with a highlighted target block. Use instanced or merged geometry so it stays above 30 fps.",
     "rubric": "Does a textured/coloured voxel terrain render; do WASD/mouse-look/jump work with collision; can blocks be placed and "
               "removed; is performance acceptable; any runtime errors?"},
    {"id": "hexagon-balls", "name": "Bouncing balls in a rotating hexagon", "html": True, "max_tokens": 3500, "system": BENCH_SYSTEM,
     "prompt": "Write a canvas animation: 20 balls of different colours and sizes bouncing inside a slowly rotating hexagon. "
               "Balls are affected by gravity and friction, collide with each other elastically, and collide correctly with the "
               "moving hexagon walls (no ball may escape). Add a slider for rotation speed and a reset button.",
     "rubric": "Physics plausibility (gravity, elastic collisions), no balls escaping the rotating hexagon, working slider and reset, "
               "smooth animation, clean code."},
    {"id": "pelican-svg", "name": "Pelican on a bicycle (SVG)", "html": True, "max_tokens": 2500, "system": BENCH_SYSTEM,
     "prompt": "Draw a pelican riding a bicycle as an inline SVG (viewBox 0 0 600 400). The pelican must be recognisable "
               "(large beak with pouch, body, wings on the handlebars, feet on the pedals); the bicycle needs two spoked wheels, "
               "a frame, handlebars, saddle and pedals. Animate the wheels turning with CSS.",
     "rubric": "Is the pelican recognisable as a pelican; is the bicycle a plausible bicycle with correct proportions; does the "
               "wheel animation work; is the SVG well structured?"},
    {"id": "todo-app", "name": "To-do app", "html": True, "max_tokens": 3500, "system": BENCH_SYSTEM,
     "prompt": "Build a polished to-do app: add, edit inline, complete, delete and reorder by drag-and-drop; filters for all/active/"
               "completed; a counter of items left; keyboard shortcuts (Enter to add, Escape to cancel editing); persistence in "
               "localStorage; light and dark themes following prefers-color-scheme with a manual toggle.",
     "rubric": "Do all listed features work (CRUD, drag reorder, filters, counter, shortcuts, persistence, theme toggle); accessible "
               "markup; no runtime errors; code quality."},
]
JUDGE_SYSTEM = ("You are a strict senior reviewer grading code produced by another model for a benchmark. Score 1-10 where 10 is "
                "production quality and fully meets the spec, 5 is partially working, 1 is unusable. Reply with JSON only: "
                '{"score": <1-10>, "reason": "<one sentence>"}.')


def bench_load():
    try:
        with open(BENCH_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"runs": [], "custom_prompts": []}


def bench_save(data):
    tmp = BENCH_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, BENCH_FILE)


def bench_prompts():
    return BENCH_PROMPTS + list(bench_load().get("custom_prompts", []))


def _bench_env(hw, entry):
    env = dict(os.environ, LLAMA_CACHE=MODEL_DIR)
    info = build_info().get("info") or {}
    if info.get("backend") == "sycl":
        env.update(oneapi_env(info.get("oneapi_root") or ""))
    env.update(server_env(hw, entry, {}))
    return env


def _device_env(hw, ids):
    """Environment that restricts the backend to the given device ordinals (subset of hw['all_compute_gpus'])."""
    be = hw.get("backend")
    if be == "cuda":
        return {"CUDA_VISIBLE_DEVICES": ",".join(map(str, ids))}
    if be == "vulkan":
        return {"GGML_VK_VISIBLE_DEVICES": ",".join(map(str, ids))}
    if be == "sycl":
        return {"ONEAPI_DEVICE_SELECTOR": ";".join(f"level_zero:{i}" for i in ids)}
    return {}


def _run_llama_bench(job, hw, entry, model, cfg, reps=3, timeout=1800, extra_env=None):
    """One llama-bench invocation; cfg may hold comma lists for one dimension. Returns list of result dicts."""
    exe = os.path.join(BIN_DIR, "llama-bench" + EXE)
    if not os.path.exists(exe):
        raise RuntimeError("bin/llama-bench not built yet - run the build first")
    cores = hw["cpu"].get("cores") or 8
    cmd = [exe, "-m", model, "-p", "512", "-n", "128", "-r", str(reps), "-ngl", "99", "-b", "2048",
           "-t", str(cores), "-o", "json",
           "-fa", str(cfg.get("fa", 1)), "-ctk", cfg.get("ctk", "f16"), "-ctv", cfg.get("ctv", "f16"),
           "-ub", str(cfg.get("ub", 512))]
    ngpu = cfg.get("ngpu") or len(hw.get("compute_gpus") or [])
    if ngpu > 1:
        cmd += ["-sm", cfg.get("sm", "layer"), "-mg", "0"]
    elif hw.get("backend") == "sycl":
        cmd += ["-sm", "none", "-mg", "0"]
    if cfg.get("ncmoe"):
        cmd += ["-ncmoe", str(cfg["ncmoe"])]
    env = _bench_env(hw, entry)
    env.update(extra_env or {})
    job.log("$ " + " ".join(os.path.basename(c) if c == exe else c for c in cmd) + (f"   [{ngpu} GPU(s)]" if extra_env else ""))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                         cwd=BIN_DIR, encoding="utf-8", errors="replace",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=not IS_WIN)
    job.proc = p
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(p)
        raise RuntimeError("llama-bench timed out")
    if job.status != "running":
        raise RuntimeError("cancelled")
    if p.returncode != 0:
        tail = "\n".join((err or out).strip().splitlines()[-4:])
        job.log(f"   ! llama-bench exit {p.returncode}: {tail}")
        return []
    try:
        start = out.index("[")
        rows = json.loads(out[start:out.rindex("]") + 1])
    except Exception:
        job.log("   ! could not parse llama-bench output")
        return []
    res = {}
    for r in rows:
        key = (r.get("type_k"), r.get("type_v"), r.get("n_ubatch"), int(bool(r.get("flash_attn"))),
               r.get("split_mode"), r.get("n_cpu_moe", 0))
        d = res.setdefault(key, {"ctk": r.get("type_k"), "ctv": r.get("type_v"), "ub": r.get("n_ubatch"),
                                 "fa": int(bool(r.get("flash_attn"))), "sm": r.get("split_mode"),
                                 "ncmoe": r.get("n_cpu_moe", 0) or 0, "ngpu": ngpu, "pp": None, "tg": None})
        if r.get("n_prompt"):
            d["pp"] = round(r.get("avg_ts") or 0, 1)
        if r.get("n_gen"):
            d["tg"] = round(r.get("avg_ts") or 0, 1)
    out_rows = list(res.values())
    for d in out_rows:
        job.log(f"   {d['ngpu']}gpu {d['sm'] or '-':6} fa={d['fa']} K={d['ctk']} V={d['ctv']} ub={d['ub']}"
                f"{' ncmoe=' + str(d['ncmoe']) if d['ncmoe'] else ''}   pp512 {d['pp']} t/s   tg128 {d['tg']} t/s")
    return out_rows


def _wait_ready(job, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if job.status != "running":
            raise RuntimeError("cancelled")
        if not SERVER.running():
            raise RuntimeError("llama-server exited while loading:\n" + "\n".join(SERVER.lines[-6:]))
        if SERVER.health()["state"] == "ready":
            return
        time.sleep(1.5)
    raise RuntimeError("llama-server did not become ready in time")


def _chat_stream(port, messages, max_tokens, timeout=1800):
    """Streamed chat completion -> (text, ttft_s, gen_s, completion_tokens)."""
    data = json.dumps({"messages": messages, "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; text = []; ntok = 0; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    if t_first is None:
                        t_first = time.time()
                    text.append(delta); ntok += 1
    t_end = time.time()
    if usage and usage.get("completion_tokens"):
        ntok = usage["completion_tokens"]
    ttft = (t_first or t_end) - t0
    return "".join(text), round(ttft, 2), round(max(1e-6, t_end - (t_first or t_end)), 2), ntok


def _slot_test(job, port, n, max_tokens=256):
    """n concurrent requests; returns aggregate tokens/s and mean per-request tokens/s."""
    prompt = "Write a detailed, numbered 25-step guide to setting up a home network, two sentences per step."
    results = []; lock = threading.Lock()

    def one():
        try:
            _, ttft, gen, ntok = _chat_stream(port, [{"role": "user", "content": prompt}], max_tokens)
            with lock:
                results.append((ttft, gen, ntok))
        except Exception as e:   # noqa
            with lock:
                results.append(None)
    t0 = time.time()
    th = [threading.Thread(target=one, daemon=True) for _ in range(n)]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    ok = [r for r in results if r]
    if not ok:
        return None
    total = sum(r[2] for r in ok)
    return {"slots": n, "agg_tps": round(total / wall, 1), "per_req_tps": round(sum(r[2] / r[1] for r in ok) / len(ok), 1),
            "ttft": round(sum(r[0] for r in ok) / len(ok), 2), "ok": len(ok), "wall": round(wall, 1)}


def _extract_html(text):
    m = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.S | re.I)
    body = m.group(1) if m else text
    if re.search(r"<!doctype html|<html", body, re.I) or re.search(r"<(svg|canvas|div|body)[\s>]", body, re.I):
        return body.strip()
    return None


def job_bench(job, body):
    """Full benchmark: sweep -> best settings -> prompt suite -> judge. Everything is appended to bench-results.json."""
    hw = detect_all(body.get("backend"), gpu_selection_arg=body.get("gpus"))
    model = body.get("model")
    if not model or not os.path.exists(model):
        raise RuntimeError("model file not found")
    entry = catalog_for(body.get("repo") or "")
    name = os.path.basename(model)
    moe = "moe" in (entry.get("arch") or "").lower() or bool(re.search(r"A\d+B|moe|gpt-oss|mixtral", name, re.I))
    if moe and "moe" not in (entry.get("arch") or "").lower():
        entry = dict(entry, arch="MoE (detected from file name)")
    stages = body.get("stages") or {}
    use = hw.get("compute_gpus") or []
    be = hw.get("backend")
    ngpu = len(use)
    job.log(f"== Benchmark {name}  backend={be}  GPUs={ngpu} ({', '.join(g['name'] for g in use)})  {'MoE' if moe else 'dense'}")
    if ngpu > 1:
        job.log(f"== {ngpu} cards: every config runs split across all of them (-sm layer/tensor)")
    run = {"id": uuid.uuid4().hex[:8], "time": time.strftime("%Y-%m-%d %H:%M"), "model": name, "path": model,
           "backend": be, "gpus": [g["name"] for g in use], "moe": moe, "sweep": [], "slots": [], "best": None,
           "prompts": [], "judge": body.get("judge") or "", "hostname": socket.gethostname()}
    msize = model_size_gb(model)
    fits = msize <= hw["vram_total_gb"] * 0.9

    # ---- sweep ----------------------------------------------------------------
    base = {"fa": 0 if (be == "sycl" and moe) else 1, "ctk": "f16", "ctv": "f16",
            "ub": 1024 if moe else (2048 if be in ("sycl", "vulkan") else 512), "sm": "layer", "ncmoe": 0}
    if not fits and moe and stages.get("offload", True):
        job.log(f"== {msize:.1f} GB model > {hw['vram_total_gb']} GB VRAM: finding the smallest --n-cpu-moe that loads")
        for n in (4, 8, 12, 16, 24, 32, 48):
            rows = _run_llama_bench(job, hw, entry, model, dict(base, ncmoe=n), reps=2)
            if rows:
                base["ncmoe"] = n; run["sweep"] += rows; break
        if not base["ncmoe"]:
            raise RuntimeError("model does not load even with 48 layers of experts on CPU")
    elif not fits:
        raise RuntimeError(f"{msize:.1f} GB model does not fit {hw['vram_total_gb']} GB of VRAM (dense models cannot offload experts)")

    def best_of(rows, key="tg"):
        rows = [r for r in rows if r.get(key)]
        return max(rows, key=lambda r: (r[key], r.get("pp") or 0)) if rows else None

    dev_env = {}
    if ngpu > 1 and stages.get("ngpu", True):
        # Stage 0: how many cards? Few-active-parameter MoE models often decode faster on fewer cards (one PCIe hop
        # per layer boundary), so try the smallest set that fits, a middle size, and all of them.
        per_card = min(g["vram_gb"] for g in use)
        need = max(1, int(-(-(msize * 1.15 + 2) // per_card)))
        cands = sorted({min(need, ngpu), min(need + 1, ngpu), ngpu}) if base["ncmoe"] == 0 else [ngpu]
        job.log(f"== Stage 0: GPU count ({msize:.1f} GB model, {per_card} GB per card -> trying {cands})")
        rows = []
        for n in cands:
            ids = hw["device_ids"][:n]
            r = _run_llama_bench(job, hw, entry, model, dict(base, ngpu=n), reps=2,
                                 extra_env=_device_env(hw, ids) if n < ngpu else None)
            rows += r
        run["sweep"] += rows
        b = best_of(rows)
        if b and b["ngpu"] < ngpu:
            base["ngpu"] = b["ngpu"]; dev_env = _device_env(hw, hw["device_ids"][:b["ngpu"]])
            job.log(f"   -> {b['ngpu']} card(s) decode fastest ({b['tg']} t/s); the remaining cards stay free for another model")
        else:
            base["ngpu"] = ngpu
    if base.get("ngpu", ngpu) > 1 and stages.get("split", True):
        job.log("== Stage 1/5: split mode")
        modes = "layer,tensor" if be == "sycl" else "layer,row" if be == "cuda" else "layer"
        rows = _run_llama_bench(job, hw, entry, model, dict(base, sm=modes), extra_env=dev_env)
        if not rows and "," in modes:     # tensor/row unsupported in this build -> layer only
            rows = _run_llama_bench(job, hw, entry, model, dict(base, sm="layer"), extra_env=dev_env)
        run["sweep"] += rows
        b = best_of(rows)
        if b: base["sm"] = b["sm"] or "layer"
    if stages.get("fa", True):
        job.log("== Stage 2/5: flash attention")
        rows = _run_llama_bench(job, hw, entry, model, dict(base, fa="0,1"), extra_env=dev_env)
        run["sweep"] += rows
        b = best_of(rows)
        if b: base["fa"] = b["fa"]
    if stages.get("kv", True):
        job.log("== Stage 3/5: KV cache type")
        if base["fa"]:
            rows = _run_llama_bench(job, hw, entry, model, dict(base, ctk="f16,q8_0,q4_0", ctv="f16,q8_0,q4_0"), extra_env=dev_env)
            rows = [r for r in rows if r["ctk"] == r["ctv"]]
        else:
            rows = _run_llama_bench(job, hw, entry, model, dict(base, ctk="f16,q8_0,q4_0", ctv="f16"), extra_env=dev_env)
        run["sweep"] += rows
        b = best_of(rows)
        if b: base["ctk"], base["ctv"] = b["ctk"], b["ctv"]
    if stages.get("ub", True):
        job.log("== Stage 4/5: micro-batch")
        rows = _run_llama_bench(job, hw, entry, model, dict(base, ub="512,1024,2048"), extra_env=dev_env)
        run["sweep"] += rows
        b = best_of(rows, key="pp")    # ub is a prefill knob; take the best prefill that does not lose decode
        if b:
            tg_ref = best_of(rows)["tg"]
            cands = [r for r in rows if r.get("tg") and r["tg"] >= tg_ref * 0.97]
            base["ub"] = max(cands, key=lambda r: r.get("pp") or 0)["ub"] if cands else b["ub"]
    final = _run_llama_bench(job, hw, entry, model, base, reps=3, extra_env=dev_env) if run["sweep"] else []
    bestrow = best_of(final) or best_of(run["sweep"])
    best = dict(base, pp=bestrow["pp"] if bestrow else None, tg=bestrow["tg"] if bestrow else None,
                kv=base["ctk"] if base["ctk"] == base["ctv"] else "f16", parallel=1, spec="none",
                gpus=[g["index"] for g in use[:base.get("ngpu", ngpu)]])
    job.log(f"== Best single-stream: {best.get('ngpu', ngpu)} GPU(s) {best['sm']} fa={best['fa']} KV={best['kv']} ub={best['ub']}"
            f"{' ncmoe=' + str(best['ncmoe']) if best['ncmoe'] else ''}  ->  pp512 {best['pp']}  tg128 {best['tg']} t/s")

    # ---- slots (server) ------------------------------------------------------
    gsel = ",".join(str(i) for i in best["gpus"]) if len(best["gpus"]) < len(hw.get("all_compute_gpus") or []) else (body.get("gpus") or "")
    hw = detect_all(body.get("backend"), gpu_selection_arg=gsel)
    serve_opts = {"backend": be, "gpus": gsel, "ctx": 32768, "kv": best["kv"], "split": best["sm"],
                  "ub": best["ub"], "fa": "on" if best["fa"] else "off", "ncmoe": best["ncmoe"],
                  "host": "127.0.0.1", "port": 8089, "mtp": True, "spec": "mtp" if entry.get("mtp") else "none"}
    draft = body.get("draft") or ""

    # ---- speculative decoding (server, single stream, code prompt) -------------------
    if stages.get("spec", True):
        cands = [("none", {})]
        if entry.get("mtp"):
            cands.append(("mtp", {}))
        if draft and os.path.exists(draft):
            cands.append(("draft", {"draft": draft, "spec_kind": "draft-mtp" if re.search(r"mtp", os.path.basename(draft), re.I) else "draft"}))
            cands.append(("draft-alt", {"draft": draft, "spec_kind": "draft" if re.search(r"mtp", os.path.basename(draft), re.I) else "draft-mtp"}))
        cands.append(("ngram", {}))
        job.log(f"== Speculative decoding: {', '.join(c[0] for c in cands)}")
        if SERVER.running():
            job.log("   stopping the running llama-server for the duration of the test"); SERVER.stop()
        code_prompt = "Write a complete Python module implementing an LRU cache with TTL expiry, type hints, docstrings and 10 unit tests."
        spec_rows = []
        for name_, extra in cands:
            o = dict(serve_opts, parallel=1, spec=name_.split("-")[0], **extra)
            try:
                SERVER.start(server_args(hw, entry, model, o), model, server_env(hw, entry, o))
                _wait_ready(job, timeout=600)
                _, ttft, gen, ntok = _chat_stream(8089, [{"role": "user", "content": code_prompt}], 700)
                spec_rows.append({"spec": name_, "tok_s": round(ntok / gen, 1), "ttft": ttft, "ok": True, "kind": extra.get("spec_kind", "")})
                job.log(f"   {name_:10} {spec_rows[-1]['tok_s']} t/s")
            except Exception as e:   # noqa
                spec_rows.append({"spec": name_, "tok_s": None, "ok": False, "err": str(e).splitlines()[0][:160], "kind": extra.get("spec_kind", "")})
                job.log(f"   {name_:10} failed: {spec_rows[-1]['err']}")
            finally:
                SERVER.stop()
        run["spec"] = spec_rows
        ok = [r for r in spec_rows if r["ok"] and r["tok_s"]]
        if ok:
            top = max(ok, key=lambda r: r["tok_s"])
            base_tps = next((r["tok_s"] for r in ok if r["spec"] == "none"), None)
            if top["spec"] != "none" and base_tps and top["tok_s"] >= base_tps * 1.08:
                best["spec"] = top["spec"].split("-")[0]
                if top["spec"].startswith("draft"):
                    best["draft"] = draft; best["spec_kind"] = top["kind"]
                    serve_opts.update(draft=draft, spec_kind=top["kind"])
                serve_opts["spec"] = best["spec"]
                job.log(f"   -> {top['spec']} wins ({top['tok_s']} vs {base_tps} t/s)")
            else:
                serve_opts["spec"] = "none"
                job.log("   -> no speculative mode beats plain decoding by 8%; keeping none")
    if stages.get("slots", True):
        job.log("== Stage 5/5: parallel slots (aggregate throughput, 256-token requests)")
        if SERVER.running():
            job.log("   stopping the running llama-server for the duration of the test"); SERVER.stop()
        for n in (1, 4, 8):
            SERVER.start(server_args(hw, entry, model, dict(serve_opts, parallel=n)), model, server_env(hw, entry, serve_opts))
            try:
                _wait_ready(job)
                r = _slot_test(job, 8089, n)
            finally:
                SERVER.stop()
            if r:
                run["slots"].append(r)
                job.log(f"   slots={n}: {r['agg_tps']} t/s aggregate, {r['per_req_tps']} t/s per request, TTFT {r['ttft']} s")
        if run["slots"]:
            single = run["slots"][0]["agg_tps"]
            # recommend more slots when aggregate gains >= 40% (serving) - per-request speed still favours 1
            gain = max(run["slots"], key=lambda r: r["agg_tps"])
            best["parallel"] = gain["slots"] if gain["agg_tps"] >= single * 1.4 else 1
            best["agg_tps"] = gain["agg_tps"]
    run["best"] = best

    # ---- prompt suite ----------------------------------------------------------
    chosen = set(body.get("prompts") or [p["id"] for p in bench_prompts()])
    prompts = [p for p in bench_prompts() if p["id"] in chosen]
    if prompts and stages.get("prompts", True):
        job.log(f"== Prompt suite: {len(prompts)} prompts with the best settings")
        if SERVER.running(): SERVER.stop()
        SERVER.start(server_args(hw, entry, model, dict(serve_opts, parallel=1)), model, server_env(hw, entry, serve_opts))
        try:
            _wait_ready(job)
            for pr in prompts:
                if job.status != "running": raise RuntimeError("cancelled")
                job.log(f"   > {pr['name']}")
                msgs = ([{"role": "system", "content": pr["system"]}] if pr.get("system") else []) + [{"role": "user", "content": pr["prompt"]}]
                text, ttft, gen, ntok = _chat_stream(8089, msgs, int(pr.get("max_tokens") or 4000))
                html = _extract_html(text) if pr.get("html") else None
                rec = {"id": pr["id"], "name": pr["name"], "ttft": ttft, "gen_s": gen, "tokens": ntok,
                       "tok_s": round(ntok / gen, 1) if gen else None, "text": text, "html": html, "score": None, "reason": ""}
                run["prompts"].append(rec)
                job.log(f"     {ntok} tokens, TTFT {ttft}s, {rec['tok_s']} t/s{'' if html else '  (no HTML block found)'}")
        finally:
            SERVER.stop()

    # ---- judge ---------------------------------------------------------------------
    judge = body.get("judge") or ""
    if run["prompts"] and stages.get("judge", True) and judge:
        jpath = model if judge == "self" else judge
        jname = os.path.basename(jpath)
        job.log(f"== Judging with {jname}")
        jentry = catalog_for("") if judge != "self" else entry
        jopts = dict(serve_opts, ctx=65536, parallel=1, ncmoe=best["ncmoe"] if judge == "self" else 0, spec="none")
        SERVER.start(server_args(hw, jentry, jpath, jopts), jpath, server_env(hw, jentry, jopts))
        try:
            _wait_ready(job)
            for rec in run["prompts"]:
                if job.status != "running": raise RuntimeError("cancelled")
                pr = next(p for p in prompts if p["id"] == rec["id"])
                q = (f"TASK GIVEN TO THE MODEL:\n{pr['prompt']}\n\nRUBRIC:\n{pr.get('rubric','')}\n\n"
                     f"MODEL OUTPUT:\n{rec['text'][:24000]}\n\nGrade it. JSON only.")
                try:
                    text, *_ = _chat_stream(8089, [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": q}], 200)
                    m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text) or re.search(r"\b(10|[1-9])\b", text)
                    rec["score"] = float(m.group(1)) if m else None
                    mr = re.search(r'"reason"\s*:\s*"([^"]{0,300})', text)
                    rec["reason"] = mr.group(1) if mr else text.strip()[:200]
                except Exception as e:   # noqa
                    rec["reason"] = f"judge failed: {e}"
                job.log(f"   {rec['name']}: {rec['score']}/10 - {rec['reason']}")
        finally:
            SERVER.stop()
        run["judge"] = jname
        scored = [r["score"] for r in run["prompts"] if r.get("score") is not None]
        run["avg_score"] = round(sum(scored) / len(scored), 1) if scored else None

    data = bench_load()
    data.setdefault("runs", []).append(run)
    bench_save(data)
    job.result = run["id"]
    job.finish(True, f"== done: results saved (run {run['id']})")


class Server:
    def __init__(self):
        self.proc, self.args, self.lines, self.started, self.port = None, [], [], None, 8080
        self.model, self.env_extra = None, {}
        self.lock = threading.Lock()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, args, model, extra_env=None):
        if self.running():
            raise RuntimeError("server already running")
        exe = os.path.join(BIN_DIR, "llama-server" + EXE)
        if not os.path.exists(exe):
            raise RuntimeError("bin/llama-server not built yet")
        env = dict(os.environ, CUDA_MODULE_LOADING="LAZY", LLAMA_CACHE=MODEL_DIR)
        info = build_info().get("info") or {}
        if info.get("backend") == "sycl":
            # SYCL binaries link against the oneAPI runtime (sycl / oneMKL / oneDNN); load setvars' environment
            oa = oneapi_env(info.get("oneapi_root") or "")
            if not oa:
                raise RuntimeError("This is a SYCL build but the Intel oneAPI runtime environment could not be loaded "
                                   f"(setvars at '{info.get('oneapi_root') or oneapi_root() or 'not found'}'). "
                                   "llama-server would exit at once with a missing sycl8.dll / mkl DLL. "
                                   "Check the oneAPI install, or rebuild with the Vulkan backend.")
            env.update(oa)
            env.setdefault("ZES_ENABLE_SYSMAN", "1")                      # free-memory queries for multi-GPU
            env.setdefault("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")   # allow single allocations > 4 GiB
        env.update(extra_env or {})                                          # device pinning etc. (server_env)
        self.env_extra = dict(extra_env or {})
        self.lines, self.args, self.model, self.started = [], args, model, time.time()
        self.port = int(args[args.index("--port") + 1]) if "--port" in args else 8080
        self.proc = subprocess.Popen([exe] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, env=env, cwd=BIN_DIR, encoding="utf-8", errors="replace",
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        p = self.proc
        for line in p.stdout:
            with self.lock:
                self.lines.append(re.sub(r"\x1b\[[0-9;]*m", "", line.rstrip("\n")))
                if len(self.lines) > 5000:
                    del self.lines[:1000]
        p.wait()
        rc = p.returncode
        if rc not in (0, None):
            with self.lock:
                self.lines.append(f"!! llama-server exited with code {rc} (0x{rc & 0xFFFFFFFF:08X})")
                hint = EXIT_HINTS.get(rc & 0xFFFFFFFF)
                if hint:
                    self.lines.append(f"!! {hint}")
                elif len(self.lines) <= 2:
                    self.lines.append("!! It printed nothing - usually a missing DLL (oneAPI runtime for SYCL builds, "
                                      "or the VC++ redistributable) or a driver/GPU fault. Run bin\\llama-server.exe "
                                      "--list-devices from a shell to see the loader error.")

    def stop(self):
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def health(self):
        if not self.running():
            return {"state": "stopped"}
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                return {"state": "ready" if json.load(r).get("status") == "ok" else "loading"}
        except Exception:
            return {"state": "loading"}

    def status(self):
        h = self.health()
        return {"running": self.running(), "state": h["state"], "pid": self.proc.pid if self.running() else None,
                "exit": (None if self.running() else (self.proc.returncode if self.proc else None)),
                "args": self.args, "model": self.model, "started": self.started, "port": self.port, "env": self.env_extra,
                "lines": len(self.lines)}


SERVER = Server()


def lan_addresses():
    """IPv4 addresses of this machine (per interface), hostname first. Best effort, no third-party modules."""
    addrs = []
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in addrs:
                addrs.append(ip)
    except Exception:
        pass
    try:   # the interface that routes to the LAN/default gateway comes first
        sck = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sck.connect(("10.255.255.255", 1))
        ip = sck.getsockname()[0]; sck.close()
        if not ip.startswith("127.") and ip in addrs:
            addrs.remove(ip)
        if not ip.startswith("127."):
            addrs.insert(0, ip)
    except Exception:
        pass
    return {"hostname": socket.gethostname(), "ips": addrs}


def firewall_allow(port, name):
    """Windows: open an inbound TCP port (prompts for elevation). Returns (ok, message)."""
    if not IS_WIN:
        return False, "only needed on Windows (use ufw/firewalld on Linux)"
    rule = f"AIOS {name} {port}"
    ps = (f"Start-Process netsh -Verb RunAs -Wait -ArgumentList 'advfirewall','firewall','add','rule',"
          f"'name=\"{rule}\"','dir=in','action=allow','protocol=TCP','localport={port}','profile=private,domain'")
    rc, out, err = run(["powershell", "-NoProfile", "-Command", ps], timeout=120)
    return rc == 0, (err or out or "rule added (if you approved the UAC prompt)")


def free_disk_gb(path):
    try:
        return gb(shutil.disk_usage(path).free)
    except Exception:
        return None


def set_models_dir(path):
    """Point AIOS at another models folder (another disk, a NAS share with GGUFs already on it). Persisted in aios-state.json."""
    global MODEL_DIR
    path = os.path.abspath(os.path.expandvars(path.strip().strip('"')))
    os.makedirs(path, exist_ok=True)
    MODEL_DIR = path
    st = load_state(); st["models_dir"] = path; save_state(st)
    return path


def local_models():
    out = []
    if os.path.isdir(MODEL_DIR):
        for root, _, files in os.walk(MODEL_DIR):
            for fn in sorted(files):
                if fn.lower().endswith(".gguf"):
                    p = os.path.join(root, fn)
                    out.append({"name": fn, "path": p, "size_gb": gb(os.path.getsize(p)),
                                "mmproj": "mmproj" in fn.lower(), "draft": bool(re.search(r"mtp|draft", fn, re.I)),
                                "shard": bool(re.search(r"-\d{5}-of-\d{5}", fn)),
                                "mtime": os.path.getmtime(p)})
    return out


def catalog_for(repo):
    return next((c for c in CATALOG if c["repo"].lower() == repo.lower()), None) or {
        "id": "custom", "name": repo.split("/")[-1].replace("-GGUF", ""), "repo": repo, "params_b": None,
        "arch": "unknown", "kv_kb_per_token": 128, "max_ctx": 131072, "mtp": False, "vision": False, "blurb": "Custom repo"}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "AIOS/" + APP_VERSION

    def log_message(self, fmt, *args):  # quieter console
        if "/api/jobs/" in (args[0] if args else "") or "/api/server/status" in (args[0] if args else ""):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def sse_lines(self, get_lines, is_done, offset):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                lines = get_lines()
                if offset < len(lines):
                    chunk = lines[offset:]
                    offset = len(lines)
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                if is_done():
                    self.wfile.write(b"event: done\ndata: {}\n\n")
                    self.wfile.flush()
                    return
                time.sleep(0.4)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    # -- routes
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        p = u.path
        try:
            if p == "/":
                body = UI_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif p == "/api/hardware":
                self.send_json(detect_all(q.get("backend", [None])[0], fresh=q.get("fresh", ["0"])[0] == "1", gpu_selection_arg=q.get("gpus", [None])[0]))
            elif p == "/api/releases":
                rels = llama_releases()
                self.send_json({"releases": rels if isinstance(rels, list) else [],
                                "error": rels.get("error") if isinstance(rels, dict) else None,
                                "releases_url": f"https://github.com/{LLAMA_REPO}/releases"})
            elif p == "/api/catalog":
                hw = detect_all(q.get("backend", [None])[0], gpu_selection_arg=q.get("gpus", [None])[0])
                self.send_json({"catalog": recommend_models(hw, int(q.get("ctx", ["32768"])[0]), q.get("kv", ["q8_0"])[0], q.get("priority", ["speed"])[0]),
                                "machine": {"vram_total_gb": hw["vram_total_gb"], "cards": len(hw.get("compute_gpus") or []), "ram_gb": hw["ram"]["total_gb"], "backend": hw["backend"]}})
            elif p == "/api/hf/files":
                repo = q.get("repo", [""])[0].strip().strip("/")
                if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
                    return self.send_json({"error": "repo must look like owner/name"}, 400)
                hw = detect_all(q.get("backend", [None])[0], gpu_selection_arg=q.get("gpus", [None])[0])
                entry = catalog_for(repo)
                files = hf_files(repo)
                ctx = int(q.get("ctx", ["65536"])[0]); kv = q.get("kv", ["q8_0"])[0]
                prio = q.get("priority", ["speed"])[0]
                files["recommendation"] = recommend(hw, entry, files["quants"], ctx, kv, prio)
                files["entry"] = entry
                self.send_json(files)
            elif p == "/api/models/local":
                self.send_json({"models": local_models(), "dir": MODEL_DIR, "free_gb": free_disk_gb(MODEL_DIR)})
            elif p == "/api/jobs":
                self.send_json({"jobs": [j.to_dict() for j in sorted(JOBS.values(), key=lambda j: j.started, reverse=True)]})
            elif p.startswith("/api/jobs/") and p.endswith("/stream"):
                j = JOBS.get(p.split("/")[3])
                if not j:
                    return self.send_json({"error": "no such job"}, 404)
                off = int(q.get("offset", ["0"])[0])
                self.sse_lines(lambda: j.lines, lambda: j.status != "running", off)
            elif p.startswith("/api/jobs/"):
                j = JOBS.get(p.split("/")[3])
                if not j:
                    return self.send_json({"error": "no such job"}, 404)
                d = j.to_dict(); d["log"] = j.lines[-400:]
                self.send_json(d)
            elif p == "/api/server/status":
                self.send_json(SERVER.status())
            elif p == "/api/server/stream":
                off = int(q.get("offset", ["0"])[0])
                self.sse_lines(lambda: SERVER.lines, lambda: not SERVER.running(), off)
            elif p == "/api/server/args":
                hw = detect_all(q.get("backend", [None])[0], gpu_selection_arg=q.get("gpus", [None])[0])
                repo = q.get("repo", [""])[0]
                entry = catalog_for(repo) if repo else CATALOG[0]
                opts = {k: v[0] for k, v in q.items()}
                opts["mtp"] = opts.get("mtp", "1") == "1"
                self.send_json({"args": server_args(hw, entry, q.get("model", ["<model>"])[0], opts),
                                "env": server_env(hw, entry, opts)})
            elif p == "/api/bench/results":
                d = bench_load()
                runs = d.get("runs", [])
                rid = q.get("id", [None])[0]
                if rid:
                    r = next((x for x in runs if x["id"] == rid), None)
                    return self.send_json(r or {"error": "no such run"}, 200 if r else 404)
                light = [{k: v for k, v in r.items() if k != "prompts"} | {"prompt_count": len(r.get("prompts", [])),
                          "avg_score": r.get("avg_score")} for r in runs]
                self.send_json({"runs": light, "prompts": bench_prompts()})
            elif p == "/api/net":
                st = SERVER.status()
                self.send_json({**lan_addresses(), "aios_port": self.server.server_address[1],
                                "aios_bind": getattr(self.server, "bind_host", "127.0.0.1"),
                                "server_port": st["port"], "server_running": st["running"],
                                "server_bind": (st["args"][st["args"].index("--host") + 1] if "--host" in (st["args"] or []) else None),
                                "model": os.path.basename(st["model"]) if st.get("model") else None})
            elif p == "/api/state":
                self.send_json(load_state())
            else:
                self.send_json({"error": "not found"}, 404)
        except urllib.error.HTTPError as e:
            self.send_json({"error": f"Hugging Face returned {e.code} for that repo" + (" (private/gated? set HF_TOKEN)" if e.code in (401, 403) else "")}, 502)
        except urllib.error.URLError as e:
            self.send_json({"error": f"Could not reach huggingface.co: {e.reason}"}, 502)
        except Exception as e:  # noqa
            self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        p = urllib.parse.urlparse(self.path).path
        try:
            body = self.read_json()
            if p == "/api/build":
                if any(j.kind == "build" and j.status == "running" for j in JOBS.values()):
                    return self.send_json({"error": "a build is already running"}, 409)
                backend = body.get("backend") or "auto"
                j = start_job("build", f"Build llama.cpp ({backend})",
                              lambda job: job_build(job, backend, bool(body.get("clean")), body.get("branch") or ""))
                self.send_json({"job": j.to_dict()})
            elif p == "/api/models/dir":
                try:
                    d = set_models_dir(body.get("dir") or "")
                except Exception as e:
                    return self.send_json({"error": f"cannot use that folder: {e}"}, 400)
                self.send_json({"dir": d, "models": local_models(), "free_gb": free_disk_gb(d)})
            elif p == "/api/download":
                repo, tag = body["repo"], body["quant"]
                files = hf_files(repo)
                qd = next((x for x in files["quants"] if x["tag"] == tag), None)
                if not qd:
                    return self.send_json({"error": f"quant {tag} not in {repo}"}, 400)
                paths = list(qd["files"])
                total = qd["size"]
                if body.get("mmproj") and files["mmproj"]:
                    mm = sorted(files["mmproj"], key=lambda f: ("f16" not in f["path"].lower(), f["size"]))[0]
                    paths.append(mm["path"]); total += mm["size"]
                if body.get("draft"):
                    # MTP / draft GGUF: from this repo, or the catalog's draft_repo (Qwen3.6 keeps it in a sibling repo)
                    dfiles = files["drafts"]
                    entry = catalog_for(repo)
                    if not dfiles and entry.get("draft_repo") and entry["draft_repo"] != repo:
                        try:
                            dfiles = hf_files(entry["draft_repo"])["drafts"] or hf_files(entry["draft_repo"])["quants"] and \
                                     [{"path": qq["files"][0], "size": qq["size"]} for qq in hf_files(entry["draft_repo"])["quants"]]
                            drepo = entry["draft_repo"]
                        except Exception:
                            dfiles, drepo = [], repo
                    else:
                        drepo = repo
                    if dfiles:
                        d = sorted(dfiles, key=lambda f: ("q8_0" not in f["path"].lower(), f["size"]))[0]   # prefer Q8_0 draft
                        extra_jobs = (drepo, d["path"], d["size"])
                        if drepo == repo:
                            paths.append(d["path"]); total += d["size"]
                        else:
                            start_job("download", f"Download draft {drepo}", lambda job, r=drepo, pth=d["path"], sz=d["size"]: job_download(job, r, [pth], sz))
                j = start_job("download", f"Download {repo}:{tag}", lambda job: job_download(job, repo, paths, total))
                self.send_json({"job": j.to_dict()})
            elif p == "/api/jobs/cancel":
                j = JOBS.get(body.get("id"))
                if j and j.status == "running":
                    kill_tree(j.proc)
                    j.finish(False, "!! cancelled")
                    if j.kind == "bench":
                        SERVER.stop()
                self.send_json({"ok": True})
            elif p == "/api/server/start":
                hw = detect_all(body.get("backend"), gpu_selection_arg=body.get("gpus"))
                entry = catalog_for(body.get("repo") or CATALOG[0]["repo"])
                model = body.get("model")
                if not model or not os.path.exists(model):
                    return self.send_json({"error": "model file not found"}, 400)
                args = server_args(hw, entry, model, body)
                SERVER.start(args, model, server_env(hw, entry, body))
                st = load_state(); st["last_server"] = body; save_state(st)
                time.sleep(2.0)
                if not SERVER.running():
                    tail = "\n".join(SERVER.lines[-6:])
                    return self.send_json({"error": f"llama-server exited immediately (code {SERVER.proc.returncode}).\n{tail}"}, 500)
                self.send_json(SERVER.status())
            elif p == "/api/bench/run":
                if any(j.kind in ("bench", "build") and j.status == "running" for j in JOBS.values()):
                    return self.send_json({"error": "a benchmark or build is already running"}, 409)
                j = start_job("bench", f"Benchmark {os.path.basename(body.get('model') or '')}", lambda job: job_bench(job, body))
                self.send_json({"job": j.to_dict()})
            elif p == "/api/bench/prompts":
                d = bench_load()
                pr = body.get("prompt") or {}
                if not pr.get("name") or not pr.get("prompt"):
                    return self.send_json({"error": "name and prompt are required"}, 400)
                pid = re.sub(r"[^a-z0-9]+", "-", pr["name"].lower()).strip("-") or uuid.uuid4().hex[:6]
                d.setdefault("custom_prompts", [c for c in d.get("custom_prompts", []) if c["id"] != pid])
                d["custom_prompts"].append({"id": pid, "name": pr["name"], "prompt": pr["prompt"], "html": bool(pr.get("html", True)),
                                            "max_tokens": int(pr.get("max_tokens") or 4000), "system": BENCH_SYSTEM if pr.get("html", True) else "",
                                            "rubric": pr.get("rubric") or "Does the output fully and correctly satisfy the task?"})
                bench_save(d)
                self.send_json({"prompts": bench_prompts()})
            elif p == "/api/net/firewall":
                ok, msg = firewall_allow(int(body.get("port") or 8080), body.get("name") or "llama-server")
                self.send_json({"ok": ok, "message": msg})
            elif p == "/api/server/stop":
                SERVER.stop()
                self.send_json(SERVER.status())
            elif p == "/api/server/chat":
                # proxy a single non-streaming chat completion so the UI can test the endpoint
                url = f"http://127.0.0.1:{SERVER.port}/v1/chat/completions"
                data = json.dumps({"messages": [{"role": "user", "content": body.get("prompt", "Say hello.")}],
                                   "max_tokens": int(body.get("max_tokens", 256)), "stream": False}).encode()
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=300) as r:
                    res = json.load(r)
                dt = time.time() - t0
                usage = res.get("usage", {})
                text = res["choices"][0]["message"]["content"]
                self.send_json({"text": text, "seconds": round(dt, 2), "usage": usage,
                                "tok_s": round(usage.get("completion_tokens", 0) / dt, 1) if dt else None,
                                "timings": res.get("timings")})
            elif p == "/api/state":
                st = load_state(); st.update(body); save_state(st)
                self.send_json(st)
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as e:  # noqa
            self.send_json({"error": str(e)}, 500)


# ---------------------------------------------------------------------------
# Embedded build / launch scripts. aios.py writes these next to itself when they are
# missing or older than SCRIPTS_VERSION, so aios.py alone is enough to distribute.
# ---------------------------------------------------------------------------
EMBEDDED_SCRIPTS = {
    'build-llama.ps1': r"""#Requires -Version 5.1
# aios-scripts-version: 6
<#
.SYNOPSIS
  Build llama.cpp from source on Windows, tuned to the detected CPU / RAM / GPU(s).

.DESCRIPTION
  1. Detects CPU, RAM and every GPU: NVIDIA (VRAM + compute capability via nvidia-smi),
     Intel Arc / AMD discrete cards (real VRAM from the display-class registry, since
     Win32_VideoController.AdapterRAM is 32-bit and caps at 4 GB) and integrated GPUs.
  2. Installs missing toolchain pieces via winget (winget source only - the msstore source
     fails behind TLS-inspecting proxies): Git, CMake, Ninja, Visual Studio Build Tools
     (C++ workload), NVIDIA CUDA Toolkit, the Vulkan SDK or the Intel oneAPI Base Toolkit (SYCL).
  3. Clones / updates llama.cpp (full clone - a shallow clone breaks `git pull`).
  4. Configures with the CUDA arch list that matches the installed cards
     (RTX 5060 Ti = compute 12.0 -> sm_120) and builds with Ninja.
  5. Installs the binaries to $InstallDir and writes build-info.json there.

  Re-run at any time to pull the latest llama.cpp and rebuild.

.PARAMETER SrcDir      Where to clone llama.cpp.           Default: <script dir>\llama.cpp
.PARAMETER InstallDir  Where the built binaries go.        Default: <script dir>\bin
.PARAMETER SkipInstall Do not try to install toolchain pieces (just detect).
.PARAMETER Clean       Delete the build directory before configuring.
.PARAMETER Branch      Git ref to build (default master).
.PARAMETER Backend     auto | cuda | vulkan | sycl | cpu
                       auto = cuda if an NVIDIA GPU is present; else sycl if a discrete Intel Arc card AND the
                       Intel oneAPI toolkit are present; else vulkan for any discrete GPU; else cpu.
                       sycl = Intel's native backend for Arc (llama.cpp docs/backend/SYCL.md): needs the
                       Intel oneAPI Base Toolkit (DPC++ icx compiler + oneMKL + oneDNN, installed via winget
                       if missing - several GB). Vulkan works on Arc with no extra SDK beyond the Vulkan SDK.
                       Also settable via the LLAMA_BACKEND environment variable.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\build-llama.ps1
#>
[CmdletBinding()]
param(
    [string]$SrcDir     = '',      # default: <script dir>\llama.cpp  (resolved below - $PSScriptRoot is empty in
    [string]$InstallDir = '',      # default: <script dir>\bin         param() defaults under powershell.exe -File)
    [switch]$SkipInstall,
    [switch]$Clean,
    [string]$Branch     = 'master',
    [ValidateSet('auto','cuda','vulkan','sycl','cpu')][string]$Backend = $(if ($env:LLAMA_BACKEND) { $env:LLAMA_BACKEND } else { 'auto' })
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

# Resolve the script folder in a way that works from a prompt, via -File, and when dot-sourced (PS 5.1 quirk)
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir -and $MyInvocation.MyCommand.Path) { $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $ScriptDir) { $ScriptDir = (Get-Location).Path }
if (-not $SrcDir)     { $SrcDir     = Join-Path $ScriptDir 'llama.cpp' }
if (-not $InstallDir) { $InstallDir = Join-Path $ScriptDir 'bin' }

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Info($msg) { Write-Host "    $msg" -ForegroundColor Gray }
function Write-Warn($msg) { Write-Host "    ! $msg" -ForegroundColor Yellow }

# Native commands (git, nvcc, cmake ...) write progress to stderr; under Windows PowerShell 5.1 with
# ErrorActionPreference=Stop that turns into a terminating error as soon as stderr is redirected.
# Run them with Continue and check $LASTEXITCODE ourselves instead.
function Invoke-Native([scriptblock]$Cmd, [switch]$IgnoreExit) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & $Cmd 2>&1 | ForEach-Object { "$_" }; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $prev }
    if (-not $IgnoreExit -and $code -ne 0) { throw "command failed (exit $code): $Cmd" }
}
$env:GIT_REDIRECT_STDERR = '2>&1'      # git: send progress to stdout so PowerShell never sees it as an error
$env:GIT_TERMINAL_PROMPT = '0'

# ---------------------------------------------------------------------------
# 1. Hardware detection
# ---------------------------------------------------------------------------
Write-Step 'Detecting hardware'

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$physCores = ($cpu.NumberOfCores)
$logCores  = ($cpu.NumberOfLogicalProcessors)
$ramGB     = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Info "CPU : $($cpu.Name.Trim())  ($physCores cores / $logCores threads)"
Write-Info "RAM : $ramGB GB"

# --- every display adapter, with real VRAM -----------------------------------------------------
# Win32_VideoController.AdapterRAM is a uint32 and caps at 4 GB, which is wrong for a 24 GB
# Arc Pro B60 (or a 16 GB RTX). The display-class registry keys carry the true size in
# HardwareInformation.qwMemorySize (QWORD, sometimes REG_BINARY); match them to WMI by DriverDesc.
function ConvertTo-Int64Value($v) {
    if ($null -eq $v) { return 0L }
    if ($v -is [byte[]]) {
        if ($v.Length -ge 8) { return [BitConverter]::ToInt64($v, 0) }
        if ($v.Length -ge 4) { return [int64][BitConverter]::ToUInt32($v, 0) }
        return 0L
    }
    try { return [int64]$v } catch { return 0L }
}
$regVram = @{}
$dispClass = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
Get-ChildItem $dispClass -ErrorAction SilentlyContinue | ForEach-Object {
    $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
    if ($p -and $p.DriverDesc) {
        $mem = ConvertTo-Int64Value $p.'HardwareInformation.qwMemorySize'
        if ($mem -le 0) { $mem = ConvertTo-Int64Value $p.'HardwareInformation.MemorySize' }
        if ($mem -gt 0 -and -not $regVram.ContainsKey($p.DriverDesc)) { $regVram[$p.DriverDesc] = $mem }
    }
}
$allGpus = @()
Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Microsoft Basic|Remote|Virtual|DisplayLink|ASPEED|Matrox' } | ForEach-Object {
    # (ASPEED / Matrox = the BMC's VGA on server boards such as the ASRock Rack ROMED8-2T - never a compute device)
    $name = $_.Name.Trim()
    $vendor = if ($name -match 'NVIDIA') { 'nvidia' } elseif ($name -match 'Intel') { 'intel' } elseif ($name -match 'AMD|Radeon|ATI') { 'amd' } else { 'other' }
    $mem = 0L
    if ($regVram.ContainsKey($name)) { $mem = $regVram[$name] } elseif ($_.AdapterRAM) { $mem = [int64]$_.AdapterRAM }
    # Integrated GPUs ("Intel(R) Graphics" on Core Ultra, UHD / Iris Xe, "AMD Radeon(TM) Graphics") are
    # display adapters, not compute targets. Anything discrete with a real frame buffer is a Vulkan target.
    $integrated = ($name -match '^Intel\(R\) (Graphics|UHD|Iris|HD)') -or ($name -match 'Radeon\(TM\) Graphics$') -or
                  ($vendor -ne 'nvidia' -and $mem -lt 3GB)
    $allGpus += [pscustomobject]@{
        Index = $allGpus.Count; Name = $name; Vendor = $vendor; VramMB = [int]($mem / 1MB)
        Integrated = $integrated; Driver = $_.DriverVersion
    }
}

# --- NVIDIA detail (compute capability) via nvidia-smi -------------------------------------------
$gpus = @()
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvsmi -and (Test-Path "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe")) {
    $nvsmi = Get-Item "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
}
if ($nvsmi) {
    $rows = Invoke-Native { & $nvsmi.Source --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits } -IgnoreExit
    foreach ($r in $rows) {
        $f = $r -split ',\s*'
        if ($f.Count -ge 4) {
            $gpus += [pscustomobject]@{
                Index = [int]$f[0]; Name = $f[1]; VramMB = [int]$f[2]
                ComputeCap = $f[3]; Driver = $f[4]
            }
        }
    }
}
if ($gpus.Count -eq 0) {
    # Fall back to the WMI/registry list (no compute cap available)
    $allGpus | Where-Object { $_.Vendor -eq 'nvidia' } | ForEach-Object {
        $gpus += [pscustomobject]@{ Index = $gpus.Count; Name = $_.Name; VramMB = $_.VramMB; ComputeCap = ''; Driver = $_.Driver }
    }
}
$vkGpus = @($allGpus | Where-Object { -not $_.Integrated })     # discrete cards a Vulkan build will run on
foreach ($g in $allGpus) {
    $cc = ($gpus | Where-Object { $_.Name -eq $g.Name } | Select-Object -First 1).ComputeCap
    $kind = if ($g.Integrated) { 'integrated' } else { 'discrete' }
    Write-Info ("GPU{0}: {1}  {2} GB  {3}/{4}{5}  driver={6}" -f $g.Index, $g.Name, [math]::Round($g.VramMB/1024,1), $g.Vendor, $kind, $(if ($cc) { "  cc=$cc" }), $g.Driver)
}
if ($allGpus.Count -eq 0) { Write-Warn 'No GPU detected at all.' }

# Map compute caps -> CMAKE_CUDA_ARCHITECTURES. Unknown caps fall back to a
# name-based guess; final fallback covers Ampere..Blackwell.
$archSet = New-Object System.Collections.Generic.HashSet[string]
foreach ($g in $gpus) {
    $cc = $g.ComputeCap -replace '\.', ''
    if ($cc -match '^\d{2,3}$') { [void]$archSet.Add($cc) }
    elseif ($g.Name -match 'RTX 50')      { [void]$archSet.Add('120') }
    elseif ($g.Name -match 'RTX 40')      { [void]$archSet.Add('89') }
    elseif ($g.Name -match 'RTX 30|A\d0') { [void]$archSet.Add('86') }
}
if ($gpus.Count -gt 0 -and $archSet.Count -eq 0) { '86','89','120' | ForEach-Object { [void]$archSet.Add($_) } }
$cudaArchs = ($archSet | Sort-Object) -join ';'
$needsCuda128 = $archSet -contains '120'   # Blackwell needs toolkit >= 12.8

# Intel oneAPI (DPC++ compiler + oneMKL + oneDNN) - needed for the SYCL backend
function Get-OneApiRoot {
    foreach ($r in @($env:ONEAPI_ROOT, "${env:ProgramFiles(x86)}\Intel\oneAPI", "$env:ProgramFiles\Intel\oneAPI")) {
        if ($r -and (Test-Path "$r\setvars.bat") -and (Test-Path "$r\compiler\latest\bin\icx.exe")) { return $r }
    }
    return $null
}
$intelGpus = @($vkGpus | Where-Object { $_.Vendor -eq 'intel' })     # discrete Intel Arc cards -> SYCL targets

# Backend choice: CUDA if an NVIDIA card is present; SYCL if a discrete Intel Arc card and oneAPI are both
# present; else Vulkan for any discrete GPU (Intel Arc / AMD); else CPU.
if ($Backend -eq 'auto') {
    if ($gpus.Count -gt 0) { $Backend = 'cuda' }
    elseif ($intelGpus.Count -gt 0 -and (Get-OneApiRoot)) { $Backend = 'sycl' }
    elseif ($vkGpus.Count -gt 0) { $Backend = 'vulkan' }
    else { $Backend = 'cpu' }
}
if ($Backend -eq 'sycl') {
    if ($intelGpus.Count -gt 0) {
        Write-Info "SYCL GPUs: $(($intelGpus | ForEach-Object { "$($_.Name) ($([math]::Round($_.VramMB/1024,1)) GB)" }) -join ', ')"
        Write-Info "Total SYCL VRAM: $([math]::Round((($intelGpus | Measure-Object VramMB -Sum).Sum)/1024,1)) GB across $($intelGpus.Count) card(s) - multi-card serving uses -sm layer; ONEAPI_DEVICE_SELECTOR=level_zero:N pins one card"
        if ((($intelGpus | Measure-Object VramMB -Sum).Sum) -gt ($ramGB * 1024)) { Write-Info "Note: VRAM ($([math]::Round((($intelGpus | Measure-Object VramMB -Sum).Sum)/1024)) GB) exceeds system RAM ($ramGB GB) - AIOS keeps mmap on so models larger than RAM still load straight into the cards." }
        if ($intelGpus.Count -lt $vkGpus.Count) { Write-Info 'Non-Intel discrete GPUs are ignored by the SYCL backend.' }
    } else {
        Write-Warn 'Backend sycl requested but no discrete Intel Arc GPU found - building anyway (it will run on integrated Intel graphics, if any).'
    }
}
if ($Backend -eq 'cuda' -and $gpus.Count -eq 0) { Write-Warn 'Backend cuda requested but no NVIDIA GPU found - building anyway.'; if (-not $cudaArchs) { $cudaArchs = '86;89;120' } }
Write-Info "Backend: $Backend$(if ($Backend -eq 'cuda') { "  (CUDA architectures: $cudaArchs)" })"
if ($Backend -eq 'vulkan') {
    if ($vkGpus.Count -gt 0) {
        Write-Info "Vulkan GPUs: $(($vkGpus | ForEach-Object { "$($_.Name) ($([math]::Round($_.VramMB/1024,1)) GB)" }) -join ', ')"
        Write-Info "Total Vulkan VRAM: $([math]::Round((($vkGpus | Measure-Object VramMB -Sum).Sum)/1024,1)) GB across $($vkGpus.Count) card(s); llama.cpp uses discrete GPUs only by default (GGML_VK_VISIBLE_DEVICES overrides)"
        if ((($vkGpus | Measure-Object VramMB -Sum).Sum) -gt ($ramGB * 1024)) { Write-Info "Note: VRAM ($([math]::Round((($vkGpus | Measure-Object VramMB -Sum).Sum)/1024)) GB) exceeds system RAM ($ramGB GB) - AIOS keeps mmap on so models larger than RAM still load straight into the cards." }
    } else {
        Write-Warn 'No discrete GPU found - the Vulkan build will run on integrated graphics (slow). Consider -Backend cpu.'
    }
}
if ($Backend -eq 'cpu') { Write-Warn 'Building CPU-only.' }

# ---------------------------------------------------------------------------
# 2. Toolchain
# ---------------------------------------------------------------------------
Write-Step 'Checking toolchain'

function Install-Winget($id, $extra) {
    if ($SkipInstall) { Write-Warn "Missing: $id (SkipInstall set)"; return }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw "winget not found - install '$id' manually." }
    Write-Info "Installing $id via winget ..."
    # --source winget: without it winget also queries the msstore source, which fails with
    # 0x8a15005e ("server certificate did not match") behind TLS-inspecting proxies / on some
    # networks - and then aborts the whole install instead of using the winget source.
    $wg = @('install','--id',$id,'-e','--source','winget','--accept-source-agreements','--accept-package-agreements','--silent','--disable-interactivity')
    if ($extra) { $wg += '--override'; $wg += $extra }
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & winget @wg 2>&1 | ForEach-Object { "$_" } | Out-Host; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $prev }
    if ($code -ne 0 -and $code -ne -1978335189) {   # -1978335189 = 0x8A15002B "no applicable upgrade / already installed"
        Write-Warn "winget install $id exited with $code (0x$('{0:X8}' -f $code))"
    }
    # refresh PATH for this session
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
}
function Require-Tool($name, $wingetId, $hint) {
    if (Get-Command $name -ErrorAction SilentlyContinue) { return }
    Install-Winget $wingetId
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        # winget adds the tool to the machine PATH but a running shell may not see it - look in the usual places
        $cands = @("$env:ProgramFiles\Git\cmd", "$env:ProgramFiles\CMake\bin", "$env:LOCALAPPDATA\Microsoft\WinGet\Links",
                   "$env:ProgramFiles\WinGet\Links", "$env:LOCALAPPDATA\Programs\Git\cmd")
        foreach ($c in $cands) { if (Test-Path (Join-Path $c "$name.exe")) { $env:Path = "$c;$env:Path"; break } }
    }
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) { throw "$name still not found after install. $hint" }
}

Require-Tool 'git'   'Git.Git'           'Install Git for Windows (winget install --id Git.Git --source winget) and re-run.'
Require-Tool 'cmake' 'Kitware.CMake'     'Install CMake (winget install --id Kitware.CMake --source winget) and re-run.'
Require-Tool 'ninja' 'Ninja-build.Ninja' 'Install Ninja (winget install --id Ninja-build.Ninja --source winget) and re-run.'

# Visual Studio Build Tools with the C++ workload (2022 or newer - vswhere finds any edition/version)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
function Find-VsPath {
    if (Test-Path $vswhere) {
        $p = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($p) { return ($p | Select-Object -First 1) }
    }
    foreach ($base in "${env:ProgramFiles}\Microsoft Visual Studio", "${env:ProgramFiles(x86)}\Microsoft Visual Studio") {
        foreach ($ver in '18','2022') { foreach ($ed in 'BuildTools','Community','Professional','Enterprise') {
            $c = Join-Path $base "$ver\$ed"
            if (Test-Path (Join-Path $c 'VC\Tools\MSVC')) { return $c }
        } }
    }
    return $null
}
$vsPath = Find-VsPath
if (-not $vsPath) {
    Install-Winget 'Microsoft.VisualStudio.2022.BuildTools' '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    $vsPath = Find-VsPath
    if (-not $vsPath) { throw 'Visual Studio Build Tools (C++ workload) still not found. Install "Desktop development with C++" from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and re-run.' }
}
Write-Info "MSVC : $vsPath"

# CUDA toolkit
function Get-CudaRoot {
    if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) { return $env:CUDA_PATH }
    $root = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $root) {
        $latest = Get-ChildItem $root -Directory | Where-Object { Test-Path "$($_.FullName)\bin\nvcc.exe" } |
                  Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending | Select-Object -First 1
        if ($latest) { return $latest.FullName }
    }
    return $null
}
$cudaRoot = $null
if ($Backend -eq 'cuda') {
    $cudaRoot = Get-CudaRoot
    $cudaVer  = $null
    if ($cudaRoot) {
        $nvccOut = (Invoke-Native { & "$cudaRoot\bin\nvcc.exe" --version } -IgnoreExit) -join "`n"
        if ($nvccOut -match 'release (\d+\.\d+)') { $cudaVer = [version]$Matches[1] }
    }
    if (-not $cudaRoot -or ($needsCuda128 -and $cudaVer -lt [version]'12.8')) {
        if ($cudaRoot) { Write-Warn "CUDA $cudaVer is too old for Blackwell (need >= 12.8)." }
        Install-Winget 'Nvidia.CUDA'
        $cudaRoot = Get-CudaRoot
        if (-not $cudaRoot) { throw 'CUDA Toolkit not found after install. Install from developer.nvidia.com/cuda-downloads and re-run.' }
        $nvccOut = (Invoke-Native { & "$cudaRoot\bin\nvcc.exe" --version } -IgnoreExit) -join "`n"
        if ($nvccOut -match 'release (\d+\.\d+)') { $cudaVer = [version]$Matches[1] }
    }
    Write-Info "CUDA : $cudaVer  ($cudaRoot)"
    $env:CUDA_PATH = $cudaRoot
    $env:Path = "$cudaRoot\bin;$env:Path"
}

# Vulkan SDK (Intel Arc / AMD / any GPU without CUDA)
$vulkanSdk = $null
if ($Backend -eq 'vulkan') {
    $vulkanSdk = $env:VULKAN_SDK
    if (-not $vulkanSdk -or -not (Test-Path "$vulkanSdk\Bin\glslc.exe")) {
        Install-Winget 'KhronosGroup.VulkanSDK'
        $vulkanSdk = [Environment]::GetEnvironmentVariable('VULKAN_SDK','Machine')
        if (-not $vulkanSdk) { $vulkanSdk = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName }
        if (-not $vulkanSdk) { throw 'Vulkan SDK not found after install. Install from vulkan.lunarg.com and re-run.' }
        $env:VULKAN_SDK = $vulkanSdk
    }
    $env:Path = "$vulkanSdk\Bin;$env:Path"
    Write-Info "Vulkan SDK: $vulkanSdk"
}

# Intel oneAPI (SYCL backend for Intel Arc) - llama.cpp docs/backend/SYCL.md
# llama.cpp's SYCL kernels name the Battlemage G31 die (Arc Pro B60 / B70, B580 = G21) via
# sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g31, which only exists in DPC++ >= 2025.2 -
# 2025.1 fails with "no member named 'intel_gpu_bmg_g31'". winget's Intel.OneAPI.BaseToolkit is stuck at 2025.1,
# so a too-old/missing toolkit is fixed with Intel's own offline installer of "Deep Learning Essentials"
# (DPC++ compiler + oneMKL + oneDNN + oneDPL - the smaller bundle SYCL.md recommends) installed side by side.
$minOneApi = [version]'2025.2'
$oneApiInstallerUrl = 'https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ffd3f1fb-09fa-4b43-9df3-955b0e29ada0/intel-deep-learning-essentials-2026.1.3.20_offline.exe'
function Get-OneApiVersion($root) {
    $vs = Get-ChildItem "$root\compiler" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Name -match '^(\d+\.\d+(\.\d+)?)') { try { [version]$Matches[1] } catch {} }
    }
    if ($vs) { return ($vs | Sort-Object -Descending | Select-Object -First 1) }
    return $null
}
function Install-OneApi {
    if ($SkipInstall) { Write-Warn 'Missing/outdated Intel oneAPI (SkipInstall set)'; return }
    $exe = Join-Path $env:TEMP (Split-Path $oneApiInstallerUrl -Leaf)
    if (-not (Test-Path $exe) -or (Get-Item $exe).Length -lt 100MB) {
        Write-Info "Downloading $oneApiInstallerUrl (~2 GB) ..."
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) { Invoke-Native { & curl.exe -L --fail --retry 3 -o $exe $oneApiInstallerUrl } | Out-Null }
        else { Invoke-WebRequest -Uri $oneApiInstallerUrl -OutFile $exe -UseBasicParsing }
    }
    Write-Info 'Running the Intel installer silently - approve the UAC prompt; this takes 10-20 minutes ...'
    # -a passes the rest to the installer; skip VS integration (Build Tools has no IDE to integrate with)
    $p = Start-Process -FilePath $exe -ArgumentList '-a','--silent','--eula','accept','-p=NEED_VS2022_INTEGRATION=0' -Verb RunAs -Wait -PassThru
    if ($p.ExitCode -ne 0) { Write-Warn "Intel installer exited with $($p.ExitCode) - see %LOCALAPPDATA%\Temp\intel_oneapi_*.log" }
}
$oneApiRoot = $null; $oneApiVer = $null
if ($Backend -eq 'sycl') {
    $oneApiRoot = Get-OneApiRoot
    if ($oneApiRoot) { $oneApiVer = Get-OneApiVersion $oneApiRoot }
    if (-not $oneApiRoot -or -not $oneApiVer -or $oneApiVer -lt $minOneApi) {
        if ($oneApiRoot) { Write-Warn "oneAPI DPC++ $oneApiVer is too old for the Arc B-series (Battlemage G31) SYCL kernels - need >= $minOneApi. Installing a newer toolkit next to it." }
        else { Write-Info 'Intel oneAPI (DPC++ icx compiler, oneMKL, oneDNN) not found - installing Intel Deep Learning Essentials ...' }
        Install-OneApi
        $oneApiRoot = Get-OneApiRoot
        if ($oneApiRoot) { $oneApiVer = Get-OneApiVersion $oneApiRoot }
        if (-not $oneApiRoot -or -not $oneApiVer -or $oneApiVer -lt $minOneApi) {
            throw "Intel oneAPI DPC++ >= $minOneApi not found after install (found: $oneApiVer). Install 'Intel Deep Learning Essentials' or the 'Intel oneAPI Toolkit' from https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html and re-run."
        }
    }
    Write-Info "oneAPI: $oneApiRoot  (DPC++ $oneApiVer)"
}

# ---------------------------------------------------------------------------
# 3. Source
# ---------------------------------------------------------------------------
Write-Step 'Fetching llama.cpp source'
if (-not (Test-Path (Join-Path $SrcDir '.git'))) {
    Invoke-Native { git clone --progress https://github.com/ggml-org/llama.cpp.git $SrcDir } | Out-Host
}
Push-Location $SrcDir
try {
    if (Test-Path '.git\shallow') { Write-Info 'Un-shallowing clone'; Invoke-Native { git fetch --unshallow } | Out-Host }
    Invoke-Native { git fetch --tags origin } | Out-Host
    Invoke-Native { git checkout -q $Branch } | Out-Host
    Invoke-Native { git reset -q --hard "origin/$Branch" } -IgnoreExit | Out-Null   # fails harmlessly for tags/commits
    $commit = (Invoke-Native { git rev-parse --short HEAD }) -join ''
    $tag    = (Invoke-Native { git describe --tags --always } -IgnoreExit) -join ''
    Write-Info "llama.cpp $tag ($commit)"
} finally { Pop-Location }

# ---------------------------------------------------------------------------
# 4. Configure + build (inside a VS developer environment, Ninja generator)
#    Ninja + nvcc avoids the "No CUDA toolset found" MSBuild-integration trap.
# ---------------------------------------------------------------------------
Write-Step 'Configuring'
$buildDir = Join-Path $SrcDir 'build'
if ($Clean -and (Test-Path $buildDir)) { Remove-Item $buildDir -Recurse -Force }

Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null

if ($Backend -eq 'sycl') {
    # setvars.bat puts icx / oneMKL / oneDNN on PATH, LIB, INCLUDE, CMPLR_ROOT ... - import its environment
    # into this PowerShell session (equivalent of the docs' cmd.exe "/K" setvars.bat && powershell trick).
    $envDump = & cmd.exe /s /c "`"$oneApiRoot\setvars.bat`" intel64 >nul 2>&1 && set"
    foreach ($line in $envDump) {
        if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
    if (-not (Get-Command icx -ErrorAction SilentlyContinue)) { throw "icx.exe not on PATH after running $oneApiRoot\setvars.bat - check the oneAPI install." }
    $icxVer = ((Invoke-Native { & icx --version } -IgnoreExit) -join ' ')
    if ($icxVer -match 'Version (\d+\.\d+(\.\d+)?)' -and [version]$Matches[1] -lt $minOneApi) {
        throw "setvars.bat selected DPC++ $($Matches[1]) (< $minOneApi) although $oneApiVer is installed - the 'compiler\latest' link in $oneApiRoot points at the old version. Repair/reinstall the newer toolkit and re-run."
    }
    Write-Info "icx  : $((Get-Command icx).Source)  ($($icxVer -replace '\s+',' ' -replace '^(.{0,110}).*','$1'))"
    $syclLs = Invoke-Native { & sycl-ls } -IgnoreExit
    if ($syclLs) { $syclLs | Where-Object { $_ -match 'gpu' } | ForEach-Object { Write-Info "sycl-ls: $_" } }
    if (-not ($syclLs -match 'level_zero:gpu')) { Write-Warn 'sycl-ls shows no [level_zero:gpu] device - install/refresh the Intel Arc driver (intel.com/arc drivers) before running the build output.' }
}

$cmakeArgs = @(
    '-S', $SrcDir, '-B', $buildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    "-DBUILD_SHARED_LIBS=$(if ($Backend -eq 'sycl') { 'ON' } else { 'OFF' })",   # static .exe files (SYCL: shared, as in examples/sycl/win-build-sycl.bat)
    '-DGGML_NATIVE=ON',                  # -march=native equivalent (Zen5: AVX-512, Zen3 EPYC 7443 / Arrow Lake: AVX2)
    '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=ON',
    '-DLLAMA_BUILD_SERVER=ON'
)
if ($Backend -eq 'vulkan') { $cmakeArgs += '-DGGML_VULKAN=ON' }
if ($Backend -eq 'sycl') {
    # docs/backend/SYCL.md "Windows - Option 1: FP16" - cl for C, icx (DPC++) for C++, oneMKL + oneDNN on by default
    $cmakeArgs += @('-DGGML_SYCL=ON', '-DGGML_SYCL_TARGET=INTEL', '-DGGML_SYCL_F16=ON',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=icx')
}
if ($cudaRoot) {
    $cmakeArgs += @(
        '-DGGML_CUDA=ON',
        "-DCMAKE_CUDA_ARCHITECTURES=$cudaArchs",
        "-DCMAKE_CUDA_COMPILER=$cudaRoot\bin\nvcc.exe",
        '-DGGML_CUDA_FA_ALL_QUANTS=ON',  # flash-attn kernels for q4_0/q8_0 KV cache
        '-DGGML_CUDA_GRAPHS=ON'
    )
}
# HTTPS for -hf downloads: Windows has no system OpenSSL, so llama.cpp can fetch+build BoringSSL
# (boringssl.googlesource.com) or LibreSSL (cdn.openbsd.org). Try BoringSSL, then LibreSSL, then none.
$sslVariants = @(
    @{ name = 'BoringSSL'; args = @('-DLLAMA_BUILD_BORINGSSL=ON', '-DLLAMA_BUILD_LIBRESSL=OFF') },
    @{ name = 'LibreSSL';  args = @('-DLLAMA_BUILD_BORINGSSL=OFF', '-DLLAMA_BUILD_LIBRESSL=ON') },
    @{ name = 'no HTTPS';  args = @('-DLLAMA_BUILD_BORINGSSL=OFF', '-DLLAMA_BUILD_LIBRESSL=OFF', '-DLLAMA_OPENSSL=OFF') }
)
$configured = $false
foreach ($v in $sslVariants) {
    $allArgs = $cmakeArgs + $v.args
    Write-Info "cmake ($($v.name)) $($allArgs -join ' ')"
    try {
        Invoke-Native { & cmake @allArgs } | Out-Host
        $configured = $true; $sslUsed = $v.name; break
    } catch {
        Write-Warn "configure with $($v.name) failed - $($_.Exception.Message)"
        Remove-Item (Join-Path $buildDir 'CMakeCache.txt') -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $buildDir '_deps') -Recurse -Force -ErrorAction SilentlyContinue
    }
}
if (-not $configured) { throw 'CMake configure failed with every SSL variant - see log above' }
if ($sslUsed -eq 'no HTTPS') { Write-Warn 'Built without HTTPS: -hf downloads will not work in llama-server (AIOS downloads models itself, so the GUI is unaffected).' }

Write-Step "Building with $logCores threads (this takes a while with CUDA / SYCL)"
Invoke-Native { & cmake --build $buildDir --config Release -j $logCores } | Out-Host

# ---------------------------------------------------------------------------
# 5. Install
# ---------------------------------------------------------------------------
Write-Step "Installing to $InstallDir"
New-Item -ItemType Directory -Force $InstallDir | Out-Null
Get-ChildItem (Join-Path $buildDir 'bin') -Filter '*.exe' | Copy-Item -Destination $InstallDir -Force
Get-ChildItem (Join-Path $buildDir 'bin') -Filter '*.dll' -ErrorAction SilentlyContinue | Copy-Item -Destination $InstallDir -Force
# CUDA runtime DLLs so the binaries run on machines without the toolkit
if ($cudaRoot) {
    Get-ChildItem "$cudaRoot\bin" -Include 'cudart64_*.dll','cublas64_*.dll','cublasLt64_*.dll' -Recurse -ErrorAction SilentlyContinue |
        Copy-Item -Destination $InstallDir -Force
}

[ordered]@{
    built_at    = (Get-Date).ToString('s')
    backend     = $Backend
    ssl         = $sslUsed
    llama_tag   = $tag
    llama_commit= $commit
    cuda        = "$cudaVer"
    cuda_archs  = $cudaArchs
    cpu         = $cpu.Name.Trim()
    cores       = $physCores
    ram_gb      = $ramGB
    gpus        = $(if ($Backend -eq 'cuda') { $gpus } elseif ($Backend -eq 'sycl') { $intelGpus } else { $vkGpus })   # the cards this build targets
    all_gpus    = $allGpus
    vulkan_sdk  = "$vulkanSdk"
    oneapi_root = "$oneApiRoot"    # SYCL builds need setvars.bat in the environment at run time (aios.py does this)
    oneapi_ver  = "$oneApiVer"
} | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $InstallDir 'build-info.json')

Write-Step 'Smoke test'
$cli = Join-Path $InstallDir 'llama-cli.exe'; $srv = Join-Path $InstallDir 'llama-server.exe'
if (-not (Test-Path $srv)) { throw "llama-server.exe was not produced - check the build log above" }
if (Test-Path $cli) { Invoke-Native { & $cli --version } -IgnoreExit | Out-Host }
if ($Backend -eq 'sycl') {
    $lsdev = Join-Path $InstallDir 'llama-ls-sycl-device.exe'
    if (Test-Path $lsdev) { Invoke-Native { & $lsdev } -IgnoreExit | Out-Host }
}
if ($Backend -ne 'cpu') { Invoke-Native { & $srv --list-devices } -IgnoreExit | Out-Host }
if ($Backend -eq 'sycl') { Write-Host "Note: SYCL binaries need the oneAPI runtime: run `"$oneApiRoot\setvars.bat`" first when launching llama-server by hand (AIOS does it for you)." -ForegroundColor Yellow }

Write-Host "`nDone. Binaries in $InstallDir" -ForegroundColor Green
Write-Host "Next: .\run-qwen.ps1 to download + launch Qwen3.8-27B tuned for this machine." -ForegroundColor Green
""",
    'build-llama.sh': r"""#!/usr/bin/env bash
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
""",
    'run-qwen.ps1': r"""#Requires -Version 5.1
# aios-scripts-version: 6
<#
.SYNOPSIS
  Download (if needed) and launch Qwen3.8-27B on llama-server, tuned for the
  detected GPUs. Defaults are the "max decode speed" profile for 2x RTX 5060 Ti 16 GB.

.PARAMETER Quant      GGUF quant tag from the HF repo. Default: auto (UD-Q4_K_XL for 2x16 GB, Q8_0 for >= 40 GB VRAM, UD-Q3_K_XL for one 16 GB card).
.PARAMETER Ctx        Context length (tokens). Default 65536.
.PARAMETER KvType     KV-cache type: f16 | q8_0 | q4_0. Default q8_0 (q4_0 for 256K context).
.PARAMETER Split      Multi-GPU split mode: layer | row. Default layer. Benchmark both with -Bench.
.PARAMETER NoMtp      Disable MTP speculative decoding (needed for --parallel > 1).
.PARAMETER Parallel   Concurrent request slots. MTP requires 1. Default 1.
.PARAMETER Vision     Also load the mmproj vision encoder (image input).
.PARAMETER Port       Server port. Default 8080.
.PARAMETER BindHost   Bind address. Default 0.0.0.0 (LAN reachable). Use 127.0.0.1 for local only.
.PARAMETER Bench      Run llama-bench instead of the server.
#>
[CmdletBinding()]
param(
    [string]$Quant    = 'auto',
    [int]$Ctx         = 65536,
    [ValidateSet('f16','q8_0','q4_0')][string]$KvType = 'q8_0',
    [ValidateSet('layer','row')][string]$Split = 'layer',
    [switch]$NoMtp,
    [int]$Parallel    = 1,
    [switch]$Vision,
    [int]$Port        = 8080,
    [string]$BindHost = '0.0.0.0',
    [string]$BinDir   = '',        # default <script dir>\bin     (resolved below; $PSScriptRoot is empty in
    [string]$ModelDir = '',        # default <script dir>\models  param() defaults under powershell.exe -File)
    [switch]$Bench
)
$ErrorActionPreference = 'Stop'
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir -and $MyInvocation.MyCommand.Path) { $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $ScriptDir) { $ScriptDir = (Get-Location).Path }
if (-not $BinDir)   { $BinDir   = Join-Path $ScriptDir 'bin' }
if (-not $ModelDir) { $ModelDir = Join-Path $ScriptDir 'models' }

$HF_REPO = 'unsloth/Qwen3.8-27B-GGUF'   # ships the MTP head tensors (blk.*.nextn.*) + fixed chat template

# --- binaries present? --------------------------------------------------------
$server = Join-Path $BinDir 'llama-server.exe'
$benchExe = Join-Path $BinDir 'llama-bench.exe'
if (-not (Test-Path $server)) {
    Write-Host "llama-server.exe not found in $BinDir" -ForegroundColor Red
    Write-Host "Build it first:  powershell -ExecutionPolicy Bypass -File `"$(Join-Path $ScriptDir 'build-llama.ps1')`"" -ForegroundColor Yellow
    exit 1
}

# --- hardware -----------------------------------------------------------------
$gpus = @()
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvsmi) {
    foreach ($p in "$env:SystemRoot\System32\nvidia-smi.exe", "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe") {
        if (Test-Path $p) { $nvsmi = Get-Item $p; break }
    }
}
if ($nvsmi) {
    $rows = @()
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $rows = & $nvsmi.Source --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>&1 } catch { $rows = @() } finally { $ErrorActionPreference = $prevEap }
    foreach ($r in $rows) {
        $f = "$r" -split ',\s*'
        if ($f.Count -ge 3 -and $f[0] -match '^\d+$') { $gpus += [pscustomobject]@{ Index=[int]$f[0]; Name=$f[1]; VramGB=[math]::Round([int]$f[2]/1024,1) } }
    }
}
if ($gpus.Count -eq 0) { Write-Host 'nvidia-smi found no GPUs - assuming 2x16 GB; pass -Quant to override.' -ForegroundColor Yellow; $totalVram = 32 }
else { $totalVram = ($gpus | Measure-Object VramGB -Sum).Sum }
$physCores = (Get-CimInstance Win32_Processor | Select-Object -First 1).NumberOfCores
Write-Host "GPUs: $($gpus.Count) x [$($gpus.Name -join ', ')]  total VRAM $totalVram GB   CPU cores $physCores" -ForegroundColor Cyan

# --- pick quant ---------------------------------------------------------------
# Sizes (GB): Q3_K_M 13.8 | IQ4_XS 15.7 | Q4_K_M 17.1 | Q5_K_M 19.8 | Q6_K 22.9 | Q8_0 29.0
if ($Quant -eq 'auto') {
    if     ($totalVram -ge 40) { $Quant = 'Q8_0'       }
    elseif ($totalVram -ge 26) { $Quant = 'UD-Q4_K_XL' }   # 2x16 GB: Unsloth dynamic 4-bit, ~Q4_K_M size, better quality
    elseif ($totalVram -ge 15) { $Quant = 'UD-Q3_K_XL' }   # single 16 GB card
    else                       { $Quant = 'UD-IQ2_XXS' }
}
Write-Host "Quant: $Quant   ctx: $Ctx   kv: $KvType   split: $Split   mtp: $(-not $NoMtp)" -ForegroundColor Cyan

# --- model source -------------------------------------------------------------
# llama.cpp downloads (with resume) into $env:LLAMA_CACHE on first use and reuses it after.
# Requires a build with HTTPS support (build-llama.ps1 sets LLAMA_BUILD_BORINGSSL=ON).
New-Item -ItemType Directory -Force $ModelDir | Out-Null
$env:LLAMA_CACHE = $ModelDir
$hfModel = "${HF_REPO}:$Quant"
Write-Host "Model: -hf $hfModel  (cache: $ModelDir)" -ForegroundColor Cyan

# --- launch -------------------------------------------------------------------
$env:CUDA_MODULE_LOADING = 'LAZY'
if ($Bench) {
    # llama-bench takes a file path, so use the cached GGUF (download it first via a normal run)
    function Find-Cached { Get-ChildItem $ModelDir -Recurse -Filter "*$Quant*.gguf" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch 'mmproj' } | Select-Object -First 1 }
    $cached = Find-Cached
    if (-not $cached) {
        # No GGUF yet: let llama-server download it (resumable), then stop it once it reports healthy.
        Write-Host "Downloading $hfModel via llama-server (first time only, ~17 GB)..." -ForegroundColor Yellow
        $dl = Start-Process -FilePath $server -PassThru -NoNewWindow -ArgumentList @(
            '-hf', $hfModel, '--no-mmproj', '-ngl', '0', '-c', '512', '--host', '127.0.0.1', '--port', "$Port")
        while (-not $dl.HasExited) {
            Start-Sleep -Seconds 3
            try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2; if ($h.status -eq 'ok') { break } } catch { }
        }
        if (-not $dl.HasExited) { Stop-Process -Id $dl.Id -Force; Start-Sleep -Seconds 1 }
        $cached = Find-Cached
        if (-not $cached) { throw "Download did not produce a $Quant GGUF in $ModelDir - check the llama-server output above." }
    }
    & $benchExe -m $cached.FullName -ngl 999 -fa 1 -sm $Split -p 512 -n 128 -r 3
    exit $LASTEXITCODE
}

$srvArgs = @(
    '-hf', $hfModel,
    '--jinja',                       # mandatory: Qwen3.8 chat template / stop tokens
    '-ngl', '999',                   # everything on GPU
    '-sm', $Split,                   # layer split across both cards (row = try for extra bandwidth)
    '-c', "$Ctx",
    '-fa', 'on',
    '--cache-type-k', $KvType, '--cache-type-v', $KvType,
    '-b', '2048', '-ub', '512',      # prompt-processing batch sizes
    '-t', "$physCores",              # CPU threads only matter for the tiny non-offloaded bits
    '--parallel', "$Parallel",
    '--host', $BindHost, '--port', "$Port",
    '--metrics'
)
if (-not $NoMtp) {
    if ($Parallel -ne 1) { throw 'MTP speculative decoding requires -Parallel 1 (pass -NoMtp for multi-slot).' }
    $srvArgs += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', '2')   # +33-39% decode on consumer GPUs
}
if (-not $Vision) { $srvArgs += '--no-mmproj' }   # -hf auto-downloads the vision encoder unless told not to

Write-Host "llama-server $($srvArgs -join ' ')" -ForegroundColor DarkGray
& $server @srvArgs
""",
    'run-qwen.sh': r"""#!/usr/bin/env bash
# aios-scripts-version: 6
# Download (if needed) and launch Qwen3.8-27B on llama-server, tuned for the
# detected GPUs. Defaults = "max decode speed" profile for 2x RTX 5060 Ti 16 GB.
#
# Usage: ./run-qwen.sh [options]
#   --quant <Q>       GGUF quant tag from the HF repo (default auto: UD-Q4_K_XL for 2x16 GB, Q8_0 for >=40 GB, UD-Q3_K_XL for one 16 GB card)
#   --ctx <n>         context length, default 65536
#   --kv <t>          KV cache type f16|q8_0|q4_0, default q8_0 (use q4_0 for 256K ctx)
#   --split <m>       layer|row, default layer (benchmark both with --bench)
#   --no-mtp          disable MTP speculative decoding (required for --parallel > 1)
#   --parallel <n>    request slots, default 1
#   --vision          also load the mmproj vision encoder
#   --port <p>        default 8080      --host <h>  default 0.0.0.0
#   --bench           run llama-bench instead of the server
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${BIN_DIR:-$SCRIPT_DIR/bin}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
HF_REPO="unsloth/Qwen3.8-27B-GGUF"   # ships MTP head tensors (blk.*.nextn.*) + fixed chat template

QUANT=auto; CTX=65536; KV=q8_0; SPLIT=layer; MTP=1; PARALLEL=1; VISION=0; PORT=8080; HOST=0.0.0.0; BENCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quant) QUANT="$2"; shift ;;   --ctx) CTX="$2"; shift ;;   --kv) KV="$2"; shift ;;
    --split) SPLIT="$2"; shift ;;   --parallel) PARALLEL="$2"; shift ;;
    --port) PORT="$2"; shift ;;     --host) HOST="$2"; shift ;;
    --no-mtp) MTP=0 ;;  --vision) VISION=1 ;;  --bench) BENCH=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac; shift
done

# --- hardware -----------------------------------------------------------------
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
TOTAL_VRAM=0; NAMES=()
for row in "${GPU_ROWS[@]}"; do IFS=',' read -r n v <<<"$row"; NAMES+=("${n## }"); TOTAL_VRAM=$(( TOTAL_VRAM + ${v## } / 1024 )); done
PHYS_CORES="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
printf '\033[1;36mGPUs: %d x [%s]  total VRAM %d GB   CPU cores %s\033[0m\n' "${#GPU_ROWS[@]}" "$(IFS=,; echo "${NAMES[*]}")" "$TOTAL_VRAM" "$PHYS_CORES"

# --- pick quant ---------------------------------------------------------------
# Sizes (GB): Q3_K_M 13.8 | IQ4_XS 15.7 | Q4_K_M 17.1 | Q5_K_M 19.8 | Q6_K 22.9 | Q8_0 29.0
if [[ "$QUANT" == auto ]]; then
  if   (( TOTAL_VRAM >= 40 )); then QUANT=Q8_0
  elif (( TOTAL_VRAM >= 26 )); then QUANT=UD-Q4_K_XL   # 2x16 GB: Unsloth dynamic 4-bit, ~Q4_K_M size, better quality
  elif (( TOTAL_VRAM >= 15 )); then QUANT=UD-Q3_K_XL   # single 16 GB card
  else QUANT=UD-IQ2_XXS; fi
fi
printf '\033[1;36mQuant: %s   ctx: %s   kv: %s   split: %s   mtp: %s\033[0m\n' "$QUANT" "$CTX" "$KV" "$SPLIT" "$MTP"

# --- model source -------------------------------------------------------------
# llama.cpp downloads (with resume) into $LLAMA_CACHE on first use and reuses it after.
# Requires a build with HTTPS support (build-llama.sh sets LLAMA_OPENSSL=ON + libssl-dev).
mkdir -p "$MODEL_DIR"
export LLAMA_CACHE="$MODEL_DIR"
HF_MODEL="$HF_REPO:$QUANT"
printf '\033[1;36mModel: -hf %s  (cache: %s)\033[0m\n' "$HF_MODEL" "$MODEL_DIR"

# --- launch -------------------------------------------------------------------
export CUDA_MODULE_LOADING=LAZY
if [[ $BENCH -eq 1 ]]; then
  # llama-bench takes a file path, so use the cached GGUF (download it first via a normal run)
  find_cached() { find "$MODEL_DIR" -name "*$QUANT*.gguf" ! -name '*mmproj*' 2>/dev/null | head -1; }
  CACHED="$(find_cached)"
  if [[ -z "$CACHED" ]]; then
    # No GGUF yet: let llama-server download it (resumable), then stop it once it reports healthy.
    echo "Downloading $HF_MODEL via llama-server (first time only, ~17 GB)..."
    "$BIN_DIR/llama-server" -hf "$HF_MODEL" --no-mmproj -ngl 0 -c 512 --host 127.0.0.1 --port "$PORT" &
    DL=$!
    while kill -0 "$DL" 2>/dev/null; do
      sleep 3
      curl -fs "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
    done
    kill "$DL" 2>/dev/null; wait "$DL" 2>/dev/null || true
    CACHED="$(find_cached)"
    [[ -z "$CACHED" ]] && { echo "Download did not produce a $QUANT GGUF in $MODEL_DIR - check the llama-server output above." >&2; exit 1; }
  fi
  exec "$BIN_DIR/llama-bench" -m "$CACHED" -ngl 999 -fa 1 -sm "$SPLIT" -p 512 -n 128 -r 3
fi

ARGS=(
  -hf "$HF_MODEL"
  --jinja                         # mandatory: Qwen3.8 chat template / stop tokens
  -ngl 999                        # everything on GPU
  -sm "$SPLIT"                    # layer split across both cards (row = try for extra bandwidth)
  -c "$CTX"
  -fa on
  --cache-type-k "$KV" --cache-type-v "$KV"
  -b 2048 -ub 512                 # prompt-processing batch sizes
  -t "$PHYS_CORES"                # CPU threads only matter for the tiny non-offloaded bits
  --parallel "$PARALLEL"
  --host "$HOST" --port "$PORT"
  --metrics
)
if [[ $MTP -eq 1 ]]; then
  [[ $PARALLEL -ne 1 ]] && { echo "MTP requires --parallel 1 (use --no-mtp for multi-slot)" >&2; exit 1; }
  ARGS+=(--spec-type draft-mtp --spec-draft-n-max 2)   # +33-39% decode on consumer GPUs
fi
[[ $VISION -eq 0 ]] && ARGS+=(--no-mmproj)   # -hf auto-downloads the vision encoder unless told not to

echo "llama-server ${ARGS[*]}"
exec "$BIN_DIR/llama-server" "${ARGS[@]}"
""",
}


def script_version(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(400)
        m = re.search(r"aios-scripts-version: (\d+)", head)
        return int(m.group(1)) if m else 0
    except FileNotFoundError:
        return -1


def ensure_scripts(log=print):
    """Write embedded scripts next to aios.py if missing/outdated (old copy kept as .bak)."""
    written = []
    for name, body in EMBEDDED_SCRIPTS.items():
        path = os.path.join(ROOT, name)
        v = script_version(path)
        if v >= SCRIPTS_VERSION:
            continue
        if v >= 0:
            try:
                shutil.copyfile(path, path + ".bak")
            except Exception:
                pass
        ps1 = name.endswith(".ps1")
        with open(path, "w", encoding="utf-8-sig" if ps1 else "utf-8", newline="\r\n" if ps1 else "\n") as f:
            f.write(body)
        written.append(f"{name} ({'created' if v < 0 else f'updated from v{max(v,0)}'})")
    if written:
        log("scripts: " + ", ".join(written))
    return written


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIOS</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#F3F5F8; --surface:#FFFFFF; --surface-2:#E9EEF4; --ink:#17202B; --ink-2:#4B5665; --muted:#7B8695;
  --line:#D5DCE5; --accent:#2B6CA3; --accent-ink:#1E4F7A; --accent-soft:#E2EDF8;
  --ok:#2E7D5B; --ok-soft:#E1F2EA; --warn:#B4741B; --warn-soft:#FBF1DE; --crit:#B23A3A; --crit-soft:#FAE6E6;
  --code-bg:#161C25; --code-ink:#DDE4EE; --code-dim:#8393A7;
  --bar:#2B6CA3; --bar2:#5C9BD1; --bar3:#A9C6E2; --bar-free:#D5DCE5;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0F1419; --surface:#161C24; --surface-2:#1E2630; --ink:#E6EBF2; --ink-2:#B3BDC9; --muted:#7F8B9A;
    --line:#2A3441; --accent:#6FA8DC; --accent-ink:#9CC5EA; --accent-soft:#1A2A3B;
    --ok:#5FBF93; --ok-soft:#173026; --warn:#E0A64B; --warn-soft:#2E2718; --crit:#E07A7A; --crit-soft:#331C1C;
    --code-bg:#0A0E13; --code-ink:#DDE4EE; --code-dim:#6F7E92;
    --bar:#4A8BC6; --bar2:#6FA8DC; --bar3:#3C566F; --bar-free:#2A3441;
  }
}
*{box-sizing:border-box}
[hidden]{display:none!important}
html,body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--accent)}
code,pre,.mono{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace}
button,select,input{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
h1,h2,h3{font-family:"Bricolage Grotesque","IBM Plex Sans",sans-serif;margin:0;line-height:1.15;text-wrap:balance}
.top{display:flex;align-items:center;gap:1.2rem;padding:.8rem 1.4rem;border-bottom:1px solid var(--line);background:var(--surface);position:sticky;top:0;z-index:5}
.top h1{font-size:1.35rem;font-weight:700;letter-spacing:-.01em}
.top .sub{color:var(--muted);font-size:.85rem}
.top .spacer{flex:1}
.pill{display:inline-flex;align-items:center;gap:.4rem;font-size:.75rem;font-weight:600;letter-spacing:.04em;padding:.22rem .6rem;border-radius:999px;background:var(--surface-2);color:var(--ink-2);white-space:nowrap}
.pill::before{content:"";width:.5rem;height:.5rem;border-radius:50%;background:var(--muted)}
.pill.ok{background:var(--ok-soft);color:var(--ok)}.pill.ok::before{background:var(--ok)}
.pill.warn{background:var(--warn-soft);color:var(--warn)}.pill.warn::before{background:var(--warn)}
.pill.crit{background:var(--crit-soft);color:var(--crit)}.pill.crit::before{background:var(--crit)}
.pill.busy::before{animation:blink 1s infinite}
@keyframes blink{50%{opacity:.2}}
@media (prefers-reduced-motion:reduce){.pill.busy::before{animation:none}}
.wrap{display:grid;grid-template-columns:320px minmax(0,1fr);gap:1.4rem;padding:1.4rem;max-width:1500px;margin:0 auto}
@media (max-width:980px){.wrap{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:1.1rem 1.2rem}
.panel h2{font-size:1.05rem;font-weight:700;display:flex;align-items:center;gap:.6rem}
.panel h2 .n{display:inline-flex;width:1.5rem;height:1.5rem;border-radius:50%;background:var(--accent);color:#fff;align-items:center;justify-content:center;font-size:.8rem}
.panel h2 .pill{margin-left:auto}
.stack{display:flex;flex-direction:column;gap:1.4rem}
.k{font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600}
.kv{display:grid;grid-template-columns:auto 1fr;gap:.3rem .9rem;font-size:.9rem;margin-top:.6rem}
.kv .k{padding-top:.2rem}
.kv .v{font-weight:500;overflow-wrap:anywhere}
.gpu{border:1px solid var(--line);border-radius:8px;padding:.6rem .8rem;margin-top:.6rem;font-size:.88rem}
.gpu .name{font-weight:600}
.gpu .meta{color:var(--ink-2);font-size:.8rem}
.tools{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.6rem}
.tools span{font-size:.75rem;padding:.15rem .5rem;border-radius:5px;background:var(--surface-2);color:var(--ink-2)}
.tools span.ok{background:var(--ok-soft);color:var(--ok)}
.tools span.miss{background:var(--crit-soft);color:var(--crit);text-decoration:line-through}
.row{display:flex;flex-wrap:wrap;gap:.7rem 1rem;align-items:end;margin-top:.9rem}
.field{display:flex;flex-direction:column;gap:.25rem;min-width:0}
.field label{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
select,input[type=text],input[type=number]{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.45rem .6rem;min-width:8rem}
input[type=text]{min-width:16rem}
.chk{display:flex;align-items:center;gap:.4rem;font-size:.9rem;padding:.45rem 0}
.btn{border:1px solid var(--line);background:var(--surface-2);border-radius:6px;padding:.5rem .95rem;font-weight:600;cursor:pointer}
.btn:hover{filter:brightness(.97)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.danger{background:var(--crit-soft);border-color:transparent;color:var(--crit)}
.btn:disabled{opacity:.5;cursor:not-allowed}
.log{background:var(--code-bg);color:var(--code-ink);border-radius:8px;padding:.8rem 1rem;margin-top:.9rem;height:260px;overflow:auto;font-family:"IBM Plex Mono",monospace;font-size:.78rem;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
.log .dim{color:var(--code-dim)}
.log .err{color:#F09393}
.log .ok{color:#9FD3A8}
.log:empty::before{content:"idle";color:var(--code-dim)}
.progress{height:6px;background:var(--surface-2);border-radius:3px;margin-top:.8rem;overflow:hidden}
.progress i{display:block;height:100%;background:var(--accent);width:0;transition:width .4s}
.models{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:.7rem;margin-top:.9rem}
.card{border:1px solid var(--line);border-radius:8px;padding:.75rem .85rem;cursor:pointer;background:var(--surface);display:flex;flex-direction:column;gap:.3rem}
.card:hover{border-color:var(--accent)}
.card.sel{border-color:var(--accent);background:var(--accent-soft)}
.card .t{font-weight:600}
.card .r{font-size:.75rem;color:var(--muted);font-family:"IBM Plex Mono",monospace;overflow-wrap:anywhere}
.card .b{font-size:.82rem;color:var(--ink-2)}
.card .tags{display:flex;gap:.3rem;flex-wrap:wrap}
.card .fit{font-size:.8rem;font-weight:600;padding:.2rem .5rem;border-radius:5px;background:var(--surface-2)}
.card .fit.ok{color:var(--ok);background:var(--ok-soft)}.card .fit.warn{color:var(--warn);background:var(--warn-soft)}.card .fit.crit{color:var(--crit);background:var(--crit-soft)}
.card.dim{opacity:.6}
.tag{font-size:.68rem;padding:.1rem .4rem;border-radius:4px;background:var(--surface-2);color:var(--ink-2)}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin-top:.9rem}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line);vertical-align:middle}
th{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.best td{background:var(--accent-soft);font-weight:600}
tr.nofit td{color:var(--muted)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.panel h3{font-weight:700}
.tw{overflow-x:auto}
.budget{margin-top:.9rem}
.bar{display:flex;height:26px;border-radius:5px;overflow:hidden;border:1px solid var(--line)}
.bar span{display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden}
.bar .free{color:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;font-size:.76rem;color:var(--ink-2);margin-top:.4rem}
.legend i{display:inline-block;width:.7em;height:.7em;border-radius:2px;margin-right:.35em}
.note{font-size:.85rem;color:var(--ink-2);margin-top:.6rem}
.note.warn{color:var(--warn)}
.endpoint{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-top:.8rem;font-size:.9rem}
.endpoint code{background:var(--surface-2);padding:.2rem .5rem;border-radius:5px}
textarea{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.5rem .6rem;font:inherit;color:inherit;min-height:60px;resize:vertical}
.answer{background:var(--surface-2);border-radius:8px;padding:.7rem .9rem;margin-top:.6rem;white-space:pre-wrap;font-size:.9rem;max-height:300px;overflow:auto}
.info{display:inline-flex;align-items:center;justify-content:center;width:1.05rem;height:1.05rem;border-radius:50%;border:1px solid var(--muted);color:var(--muted);font:600 .62rem/1 "IBM Plex Sans",sans-serif;text-transform:none;letter-spacing:0;margin-left:.3rem;cursor:help;position:relative;vertical-align:middle;font-style:normal}
.info::before{content:"i";font-family:"IBM Plex Serif","Georgia",serif;font-style:italic;font-weight:700;font-size:.7rem}
.info:hover,.info:focus{border-color:var(--accent);color:var(--accent);outline:none}
.info::after{content:attr(data-tip);position:absolute;left:0;top:1.4rem;z-index:20;width:19rem;max-width:70vw;background:var(--ink);color:var(--bg);padding:.55rem .7rem;border-radius:6px;font:400 .78rem/1.45 "IBM Plex Sans",sans-serif;text-transform:none;letter-spacing:0;white-space:normal;box-shadow:0 6px 20px rgba(0,0,0,.25);opacity:0;visibility:hidden;transition:opacity .12s;pointer-events:none}
.info:hover::after,.info:focus::after{opacity:1;visibility:visible}
.field:last-child .info::after,.tips-right .info::after{left:auto;right:0}
@media (prefers-reduced-motion:reduce){.info::after{transition:none}}
.toast{position:fixed;bottom:1rem;right:1rem;background:var(--ink);color:var(--bg);padding:.6rem .9rem;border-radius:8px;font-size:.88rem;opacity:0;transition:opacity .2s;pointer-events:none}
.toast.show{opacity:1}
</style>
</head>
<body>
<div class="top">
  <div><h1>AIOS <span style="font-size:.7rem;color:var(--muted);font-weight:500" id="ver"></span></h1><div class="sub">Browser OS for llama.cpp — detect, build, download, serve · <span class="mono" id="root"></span></div></div>
  <div class="spacer"></div>
  <span class="pill" id="pill-build">build: checking</span>
  <span class="pill" id="pill-server">server: stopped</span>
  <button class="btn" id="btn-refresh">Re-detect</button>
</div>

<div class="wrap">
  <!-- ===== left: machine ===== -->
  <aside class="stack">
    <div class="panel" id="hw">
      <h2>This machine <span class="pill" id="pill-hw">detecting</span></h2>
      <div class="kv" id="hw-kv"></div>
      <div id="hw-gpus"></div>
      <div class="k" style="margin-top:1rem">Toolchain</div>
      <div class="tools" id="hw-tools"></div>
      <div class="note" id="hw-note"></div>
    </div>
    <div class="panel">
      <h2>Current build</h2>
      <div class="kv" id="build-kv"><div class="k">status</div><div class="v">checking…</div></div>
    </div>
  </aside>

  <!-- ===== right: workflow ===== -->
  <main class="stack">
    <!-- 1 build -->
    <section class="panel">
      <h2><span class="n">1</span>Build llama.cpp for this hardware <span class="pill" id="pill-buildjob">idle</span></h2>
      <div class="note">Runs <code id="build-script">build-llama</code> next to aios.py: installs missing toolchain pieces, clones or updates llama.cpp, configures for the detected GPUs and installs to <code>bin/</code>.</div>
      <div class="row">
        <div class="field"><label>Backend<i class="info" tabindex="0" data-tip="Which GPU API llama.cpp is compiled against. CUDA for NVIDIA cards (fastest on the 5060 Ti). SYCL is Intel's native oneAPI backend for Arc (A/B-series incl. Arc Pro B60) - needs the Intel oneAPI Base Toolkit, installed automatically (several GB). Vulkan works on any GPU incl. Intel Arc and AMD with no extra toolkit. CPU builds run without a GPU. Auto picks CUDA when an NVIDIA card is present, SYCL when an Arc card and oneAPI are present, otherwise Vulkan."></i></label>
          <select id="backend">
            <option value="auto">auto (detected)</option>
            <option value="cuda">CUDA — NVIDIA</option>
            <option value="sycl">SYCL — Intel Arc (oneAPI)</option>
            <option value="vulkan">Vulkan — Intel Arc / AMD / any GPU</option>
            <option value="cpu">CPU only</option>
          </select></div>
        <div class="field"><label>Branch / release<i class="info" tabindex="0" data-tip="Which llama.cpp source to build. master = the newest commit (best for brand-new models like Qwen3.8). A release tag (b10758…) is a fixed snapshot you can reproduce later. Custom accepts any branch, tag or commit hash."></i> <a id="rel-link" href="https://github.com/ggml-org/llama.cpp/releases" target="_blank" rel="noopener" style="font-weight:500;text-transform:none;letter-spacing:0">(all releases ↗)</a></label>
          <div style="display:flex;gap:.4rem">
            <select id="branch-sel" style="min-width:14rem"><option value="master">master (latest commit)</option><option value="__custom">custom…</option></select>
            <input type="text" id="branch" placeholder="tag, branch or commit" style="min-width:10rem" hidden>
          </div></div>
        <label class="chk"><input type="checkbox" id="clean"> clean rebuild<i class="info" tabindex="0" data-tip="Deletes llama.cpp/build before configuring, so every file is compiled again. Use after changing backend or CUDA toolkit, or when a build behaves oddly. Normal builds are incremental and much faster."></i></label>
        <button class="btn primary" id="btn-build">Build</button>
        <button class="btn danger" id="btn-build-cancel" hidden>Cancel</button>
      </div>
      <div class="log" id="log-build"></div>
    </section>

    <!-- 2 model -->
    <section class="panel">
      <h2><span class="n">2</span>Pick and download a model <span class="pill" id="pill-dl">idle</span></h2>
      <div class="note" id="catalog-note" style="display:none"></div>
      <div class="models" id="catalog"></div>
      <div class="row">
        <div class="field" style="flex:1"><label>Or any Hugging Face GGUF repo<i class="info" tabindex="0" data-tip="owner/name of a Hugging Face repository containing .gguf files (e.g. unsloth/Qwen3.8-27B-GGUF). AIOS lists every quant in it with its real size and marks the ones that fit your VRAM. Set HF_TOKEN in the environment for gated repos."></i></label><input type="text" id="repo" placeholder="owner/Model-GGUF"></div>
        <button class="btn" id="btn-repo">List quants</button>
      </div>
      <div class="row">
        <div class="field"><label>Context (tokens)<i class="info" tabindex="0" data-tip="How many tokens the model can hold at once — prompt plus reply. More context costs VRAM for the KV cache (Qwen3.8: ~64 KB per token at f16, half that at q8_0). 65536 is plenty for coding; Qwen3.8 supports up to 262144."></i></label>
          <select id="ctx"><option>8192</option><option>16384</option><option>32768</option><option selected>65536</option><option>131072</option><option>262144</option></select></div>
        <div class="field"><label>KV cache<i class="info" tabindex="0" data-tip="Precision of the attention key/value cache that grows with context. f16 = exact, biggest. q8_0 = half the size with no practical quality loss (recommended). q4_0 = a quarter of the size, slight quality loss — use for very long contexts."></i></label>
          <select id="kv"><option>f16</option><option selected>q8_0</option><option>q4_0</option></select></div>
        <div class="field"><label>Priority<i class="info" tabindex="0" data-tip="How the recommended quant is chosen. Decode speed picks the best 4-bit quant that fits: fewer bytes per token to read from VRAM = more tokens per second. Quality picks the largest quant that fits (Q6/Q8), which is slower but closer to the original weights."></i></label>
          <select id="prio"><option value="speed" selected>decode speed (best 4-bit that fits)</option><option value="quality">quality (largest that fits)</option></select></div>
        <label class="chk"><input type="checkbox" id="want-draft"> also download MTP / draft GGUF<i class="info" tabindex="0" data-tip="The separate speculative-decoding draft (MTP head) for models that ship it as its own file (Qwen3.6-35B-A3B, Qwen3.8-Flash-Next). Small (2-4 GB). Lets the Speculative = draft option and the benchmark's speculative stage run."></i></label>
        <label class="chk"><input type="checkbox" id="want-mmproj"> also download vision encoder (mmproj)<i class="info" tabindex="0" data-tip="Downloads the separate ~0.9 GB mmproj file so the model can look at images. Only needed for vision use; it costs VRAM and startup time when loaded."></i></label>
      </div>
      <div id="rec-note" class="note"></div>
      <div class="budget" id="budget" hidden>
        <div class="bar" id="bar"></div>
        <div class="legend"><span><i style="background:var(--bar)"></i>weights</span><span><i style="background:var(--bar2)"></i>KV cache at chosen context</span><span><i style="background:var(--bar3)"></i>compute buffers / draft head</span><span><i style="background:var(--bar-free)"></i>free</span></div>
      </div>
      <div class="tw"><table id="quants" hidden><thead><tr><th>Quant<i class="info" tabindex="0" data-tip="Quantization = how many bits per weight. Q8 ≈ lossless, Q6/Q5 near-lossless, Q4 the usual sweet spot, Q3/Q2 noticeably degraded. UD- = Unsloth Dynamic (mixed precision, better than plain at the same size). K = k-quant; XL/M/S = more/less precision on important layers; IQ = importance-matrix quants."></i></th><th class="num">Size</th><th>Files</th><th>Fit<i class="info" tabindex="0" data-tip="Whether weights + KV cache at the chosen context + ~1.2 GB per GPU of buffers fit in 94% of usable VRAM. RAM offload only = would need some layers in system RAM (slow). ★ = the recommended pick for the chosen priority."></i></th><th></th></tr></thead><tbody></tbody></table></div>
      <div class="row" style="align-items:flex-end">
        <div class="field" style="flex:1"><label>Models folder<i class="info" tabindex="0" data-tip="Where GGUF files are downloaded to and served from. Point it at another disk, an existing llama.cpp / LM Studio model folder, or a NAS share (\\\\nas\\models) that already holds the models - AIOS lists every .gguf it finds there and skips downloads you already have. Also settable with --models-dir or the AIOS_MODELS environment variable."></i></label><input type="text" id="models-dir" spellcheck="false" style="min-width:24rem"></div>
        <button class="secondary" id="btn-models-dir" type="button">Use folder</button>
        <span class="note" id="models-free" style="margin:0 0 .5rem"></span>
      </div>
      <div class="progress" id="dl-progress" hidden><i></i></div>
      <div class="log" id="log-dl" style="height:150px"></div>
    </section>

    <!-- 3 serve -->
    <section class="panel">
      <h2><span class="n">3</span>Serve <span class="pill" id="pill-srv">stopped</span></h2>
      <div class="row">
        <div class="field" style="flex:1"><label>Model file (models/)<i class="info" tabindex="0" data-tip="The GGUF weights to serve, from the models folder next to aios.py. For sharded models only the first shard is listed; llama.cpp loads the rest automatically."></i></label><select id="model-file" style="min-width:20rem"></select></div>
        <div class="field"><label>Vision encoder<i class="info" tabindex="0" data-tip="Optional mmproj GGUF that enables image input on the /v1/chat/completions endpoint. Leave as none for text-only serving."></i></label><select id="mmproj-file"><option value="">none</option></select></div>
      </div>
      <div class="row">
        <div class="field"><label>Context<i class="info" tabindex="0" data-tip="Context window for the running server (-c). Reserved up front in VRAM, so bigger = less room for weights. With multiple slots the window is shared between them."></i></label><input type="number" id="s-ctx" value="65536" step="4096" style="min-width:7rem"></div>
        <div class="field"><label>KV cache<i class="info" tabindex="0" data-tip="--cache-type-k/v for the server: f16 exact, q8_0 half size (recommended on NVIDIA), q4_0 quarter size (fastest on Intel Arc / Battlemage in Warren's benchmarks; on SYCL above ~16K context use f16). Needs flash attention."></i></label><select id="s-kv"><option>f16</option><option selected>q8_0</option><option>q4_0</option></select></div>
        <div class="field"><label>GPUs<i class="info" tabindex="0" data-tip="Which cards this server uses. All = one model split across every card (-sm layer). Untick cards to leave them free for a second llama-server on another port (one model per card, llama-swap style); AIOS pins the process with CUDA_VISIBLE_DEVICES / GGML_VK_VISIBLE_DEVICES / ONEAPI_DEVICE_SELECTOR. The quant recommendation follows the ticked VRAM."></i></label><div id="s-gpus" style="display:flex;gap:.6rem;flex-wrap:wrap;padding:.35rem 0"></div></div>
        <div class="field"><label>GPU split<i class="info" tabindex="0" data-tip="How a model is divided over two or more GPUs (-sm). layer = whole layers per card, runs them in sequence; safe default and the right choice for 3-4 Arc cards. row = every matrix split across cards (CUDA only; falls back to layer on SYCL). tensor = SYCL tensor parallelism, tuned for exactly 2 cards."></i></label><select id="s-split"><option value="layer">layer</option><option value="row">row</option><option value="tensor">tensor (SYCL, 2 cards)</option></select></div>
        <div class="field"><label>GPU layers<i class="info" tabindex="0" data-tip="How many model layers go to the GPU (-ngl). 999 = all of them (fastest). Lower it only when the model does not fit in VRAM; the rest runs on the CPU from system RAM, much slower."></i></label><input type="number" id="s-ngl" value="999" style="min-width:5rem"></div>
        <div class="field"><label>Slots<i class="info" tabindex="0" data-tip="Parallel request slots (--parallel). 1 = one request at a time, fastest per request and required for MTP. More slots serve several clients at once but split the context window and VRAM between them. MoE models gain a lot of aggregate throughput from 4-8 slots (benchmark it in section 4)."></i></label><input type="number" id="s-par" value="1" min="1" style="min-width:5rem"></div>
        <div class="field"><label>Micro-batch<i class="info" tabindex="0" data-tip="-ub: tokens processed per GPU pass during prompt processing. Bigger = faster prefill on Arc (2048 ≈ +60%), but on MoE models it inflates the expert gather - 1024 is usually best there. Auto picks per backend/model; the benchmark finds the real optimum."></i></label><select id="s-ub"><option value="">auto</option><option>512</option><option>1024</option><option>2048</option></select></div>
        <div class="field"><label>Flash attn<i class="info" tabindex="0" data-tip="-fa. On = fused attention, needed for a quantized V cache. Off = AIOS default for MoE on SYCL (faster/stabler on Arc). Auto lets llama.cpp decide."></i></label><select id="s-fa"><option value="">auto (AIOS)</option><option value="on">on</option><option value="off">off</option></select></div>
        <div class="field"><label>Experts on CPU<i class="info" tabindex="0" data-tip="--n-cpu-moe N: keep the expert weights of the first N layers in system RAM (MoE models only). Lets a model bigger than VRAM run - attention and shared weights stay on the cards, experts stream from RAM. 0 = everything on the GPUs. Start around a third of the layer count and raise it until the model loads."></i></label><input type="number" id="s-ncmoe" value="0" min="0" style="min-width:5rem"></div>
        <div class="field"><label>Speculative<i class="info" tabindex="0" data-tip="Speculative decoding. MTP = the model's own draft head inside the GGUF (Qwen3.8-27B; slots must be 1). draft = a separate MTP/draft GGUF (pick it in Draft model; Qwen3.6-35B-A3B, Qwen3.8-Flash-Next). n-gram = draft-free lookup decoding (--spec-type ngram-simple) - a cheap win on repetitive code/agent output. none = off."></i></label><select id="s-spec"><option value="">auto</option><option value="mtp">MTP (built in)</option><option value="draft">draft GGUF</option><option value="ngram">n-gram</option><option value="none">none</option></select></div>
        <div class="field"><label>Draft model<i class="info" tabindex="0" data-tip="Separate MTP / draft GGUF from the models folder (-md). Used when Speculative = draft or auto."></i></label><select id="s-draft"><option value="">none</option></select></div>
        <label class="chk"><input type="checkbox" id="s-mtp" checked> MTP draft (Qwen3.8, slots=1)<i class="info" tabindex="0" data-tip="Multi-token prediction speculative decoding (--spec-type draft-mtp). Qwen3.8 carries a small draft head inside the GGUF that guesses the next tokens; the main model verifies them in one pass. Roughly +33-39% decode speed, no quality change. Only applies to models with an MTP head and needs Slots = 1."></i></label>
        <div class="field"><label>Port<i class="info" tabindex="0" data-tip="TCP port for llama-server. The OpenAI-compatible API is at http://host:port/v1, the built-in chat UI at /, Prometheus metrics at /metrics."></i></label><input type="number" id="s-port" value="8080" style="min-width:6rem"></div>
        <div class="field"><label>Bind<i class="info" tabindex="0" data-tip="Network interface to listen on. 0.0.0.0 = reachable from other machines on your LAN (OpenCode, Open WebUI, the latency monitor). 127.0.0.1 = this computer only."></i></label><select id="s-host"><option>0.0.0.0</option><option>127.0.0.1</option></select></div>
      </div>
      <div class="row">
        <div class="field" style="flex:1"><label>Extra llama-server flags<i class="info" tabindex="0" data-tip="Anything else to append to the llama-server command, e.g. sampling defaults. Qwen3.8 non-thinking mode: --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5. Thinking mode: --temp 1.0 --top-p 0.95 --top-k 20."></i></label><input type="text" id="s-extra" placeholder="e.g. --temp 0.7 --top-p 0.8"></div>
        <button class="btn primary" id="btn-start">Start server</button>
        <button class="btn danger" id="btn-stop" disabled>Stop</button>
      </div>
      <div class="note mono" id="s-cmd"></div>
      <div class="endpoint" id="endpoint" hidden>
        <div style="display:flex;gap:.8rem;align-items:center;flex-wrap:wrap"><b>Access the running model</b><span class="spacer"></span>
          <a id="ep-ui" class="btn" target="_blank" rel="noopener">Open chat in browser</a><a id="ep-metrics" target="_blank" rel="noopener" style="font-size:.85rem">metrics</a></div>
        <div id="lan-list" style="margin-top:.5rem;font-size:.9rem"></div>
        <div id="lan-note" class="note" style="margin:.5rem 0 0"></div>
        <details style="margin-top:.5rem"><summary style="cursor:pointer;font-size:.88rem">Connect a coding agent (OpenAI-compatible)</summary>
          <div class="row" style="margin-top:.4rem"><div class="field"><label>Agent</label><select id="agent-kind"><option value="opencode">OpenCode</option><option value="claude">Claude Code (via OpenAI-compatible router)</option><option value="continue">Continue / Cline / Roo</option><option value="dsh">DeepSeek Harness (dsh)</option><option value="openwebui">Open WebUI</option><option value="env">Generic: OPENAI_BASE_URL</option></select></div>
          <div class="field"><label>Address</label><select id="agent-addr"></select></div><button class="btn" id="agent-copy">Copy</button></div>
          <pre id="agent-snippet" style="margin:.4rem 0 0;font-size:.78rem"></pre>
        </details>
      </div>
      <div class="log" id="log-srv"></div>
      <div class="row"><div class="field" style="flex:1"><label>Test prompt<i class="info" tabindex="0" data-tip="Sends one chat completion to the running server through AIOS and shows the reply with tokens/s for generation and prompt processing."></i></label><textarea id="prompt">Explain what an MTP draft head does in two sentences.</textarea></div><button class="btn" id="btn-test">Send</button></div>
      <div class="answer" id="answer" hidden></div>
    </section>

    <!-- 4 benchmark -->
    <section class="panel">
      <h2><span class="n">4</span>Benchmark <span class="pill" id="pill-bench">idle</span></h2>
      <div class="note">Finds the fastest settings for <b>this machine</b> and the detected cards (all cards in parallel when there is more than one), then runs the prompts streamers use to compare models - Browser OS, a browser game, a voxel world, the classic set - with speed numbers, a live render of each result and a 1-10 score from a judge model.</div>
      <div class="row">
        <div class="field" style="flex:1"><label>Model to benchmark<i class="info" tabindex="0" data-tip="A GGUF from the models folder. MoE models that are bigger than VRAM are tried with experts in RAM (--n-cpu-moe)."></i></label><select id="b-model" style="min-width:20rem"></select></div>
        <div class="field"><label>Draft / MTP GGUF<i class="info" tabindex="0" data-tip="Optional separate MTP or draft model to test in the speculative stage (download it with the 'also download MTP / draft GGUF' box in section 2)."></i></label><select id="b-draft"><option value="">none</option></select></div>
        <div class="field"><label>Judge<i class="info" tabindex="0" data-tip="Model that grades each prompt result 1-10 against a rubric. 'same model' reuses the benchmarked model; a stronger local model gives fairer scores. none = skip judging."></i></label><select id="b-judge"><option value="self">same model</option><option value="">none</option></select></div>
      </div>
      <div class="row" style="flex-wrap:wrap;gap:.8rem">
        <label class="chk"><input type="checkbox" id="bs-ngpu" checked> GPU count</label>
        <label class="chk"><input type="checkbox" id="bs-split" checked> split mode</label>
        <label class="chk"><input type="checkbox" id="bs-fa" checked> flash attention</label>
        <label class="chk"><input type="checkbox" id="bs-kv" checked> KV cache type</label>
        <label class="chk"><input type="checkbox" id="bs-ub" checked> micro-batch</label>
        <label class="chk"><input type="checkbox" id="bs-slots" checked> parallel slots</label>
        <label class="chk"><input type="checkbox" id="bs-offload" checked> expert offload (MoE &gt; VRAM)</label>
        <label class="chk"><input type="checkbox" id="bs-spec" checked> speculative (MTP / draft / n-gram)</label>
        <label class="chk"><input type="checkbox" id="bs-prompts" checked> prompt suite</label>
        <label class="chk"><input type="checkbox" id="bs-judge" checked> judge</label>
      </div>
      <div class="row" style="flex-wrap:wrap;gap:.6rem" id="b-prompts"></div>
      <details style="margin:.6rem 0"><summary style="cursor:pointer;color:var(--muted);font-size:.85rem">Add your own prompt</summary>
        <div class="row"><div class="field"><label>Name</label><input type="text" id="bp-name" placeholder="e.g. Chess in the browser"></div><div class="field"><label>Max tokens</label><input type="number" id="bp-max" value="4000" style="min-width:6rem"></div><label class="chk"><input type="checkbox" id="bp-html" checked> expects HTML (render preview)</label></div>
        <div class="row"><div class="field" style="flex:1"><label>Prompt</label><textarea id="bp-prompt" rows="3"></textarea></div></div>
        <div class="row"><div class="field" style="flex:1"><label>Rubric for the judge</label><input type="text" id="bp-rubric" placeholder="What must be true for a 10/10?"></div><button class="btn" id="bp-save">Save prompt</button></div>
      </details>
      <div class="row">
        <button class="btn primary" id="btn-bench">Run benchmark</button>
        <button class="btn danger" id="btn-bench-cancel" disabled>Cancel</button>
        <span class="note" style="margin:0">Full run: 15-30 min of sweep plus the prompts (several minutes each) and the judge. The running llama-server is stopped during the slot test and the prompt suite.</span>
      </div>
      <div class="log" id="log-bench" style="height:200px"></div>
      <div id="bench-out"></div>
      <h3 style="margin-top:1.2rem;font-size:.95rem">Previous runs</h3>
      <div class="tw"><table id="bench-runs"><thead><tr><th>When</th><th>Model</th><th>Backend / GPUs</th><th>Best settings</th><th class="num">tg128</th><th class="num">pp512</th><th class="num">Slots agg.</th><th class="num">Score</th><th></th></tr></thead><tbody></tbody></table></div>
    </section>
  </main>
</div>
<div class="toast" id="toast"></div>

<script>
const $=s=>document.querySelector(s);
const api=async(p,opt)=>{const r=await fetch(p,opt);const j=await r.json();if(!r.ok||j.error)throw new Error(j.error||r.statusText);return j;};
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
const toast=m=>{const t=$('#toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),3500);};
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmtGB=g=>g>=1?`${g.toFixed(g>=10?1:2)} GB`:`${Math.round(g*1024)} MB`;
let HW=null, CATALOG=[], SEL=null, QUANTS=null, LOCAL=[];

/* ---------- log streaming ---------- */
function stream(url,el,onDone){
  const es=new EventSource(url);
  es.onmessage=e=>{const lines=JSON.parse(e.data);for(const l of lines){const d=document.createElement('div');
    d.className=/^!!|error|fatal|failed/i.test(l)?'err':/^==|OK$|finished/i.test(l)?'ok':/^\$ |^    /.test(l)?'dim':'';d.textContent=l;el.appendChild(d);}
    el.scrollTop=el.scrollHeight;};
  es.addEventListener('done',()=>{es.close();onDone&&onDone();});
  es.onerror=()=>{es.close();onDone&&onDone();};
  return es;
}
function setPill(id,text,cls){const p=$(id);p.textContent=text;p.className='pill '+(cls||'');}

/* ---------- LAN access ---------- */
let NET=null;
async function loadNet(){
  try{NET=await api('/api/net');}catch{return;}
  const port=NET.server_port,lan=NET.server_bind==='0.0.0.0';
  const addrs=lan?[...NET.ips.map(ip=>`http://${ip}:${port}`),`http://${NET.hostname}:${port}`]:[`http://127.0.0.1:${port}`];
  $('#lan-list').innerHTML=addrs.map(a=>`<div style="display:flex;gap:.6rem;align-items:center;margin:.15rem 0"><code>${esc(a)}/v1</code><button class="btn" data-copy="${esc(a)}/v1" style="padding:.1rem .5rem;font-size:.75rem">copy</button><a href="${esc(a)}/" target="_blank" rel="noopener" style="font-size:.8rem">chat</a></div>`).join('');
  $('#lan-list').querySelectorAll('[data-copy]').forEach(b=>b.onclick=()=>{navigator.clipboard.writeText(b.dataset.copy);toast('Copied');});
  if(lan){$('#lan-note').innerHTML=`Reachable from other machines on your network (bound to 0.0.0.0). If a client gets <i>connection refused</i>, Windows Firewall is usually blocking the port: <button class="btn" id="fw-btn" style="padding:.1rem .5rem;font-size:.75rem">Allow port ${port} inbound</button> (asks for admin).${NET.aios_bind!=='0.0.0.0'?` The AIOS page itself is local-only; start it with <code>--host 0.0.0.0</code> to reach it from the LAN too.`:''}`;
    $('#fw-btn').onclick=async()=>{try{const r=await post('/api/net/firewall',{port,name:'llama-server'});toast(r.ok?'Firewall rule added':r.message);}catch(e){toast(e.message);}};}
  else{$('#lan-note').innerHTML=`Bound to 127.0.0.1: only this computer can reach the model. Set <b>Bind</b> to 0.0.0.0 and restart the server to use it from agents on other machines.`;}
  const sel=$('#agent-addr');sel.innerHTML=addrs.map(a=>`<option>${esc(a)}</option>`).join('');renderAgent();
}
function renderAgent(){
  if(!NET)return;const base=$('#agent-addr').value+'/v1',model=NET.model||'local',kind=$('#agent-kind').value;let t='';
  if(kind==='opencode')t=`// ~/.config/opencode/opencode.json  (Windows: %USERPROFILE%\\.config\\opencode\\opencode.json)
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "aios": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "AIOS llama-server",
      "options": { "baseURL": "${base}", "apiKey": "none" },
      "models": { "${model}": { "name": "${model}" } }
    }
  },
  "model": "aios/${model}"
}`;
  else if(kind==='claude')t=`# Claude Code talks the Anthropic API, so put an OpenAI->Anthropic router in front of llama-server,
# e.g. claude-code-router (npm i -g @musistudio/claude-code-router) with this provider:
#   { "name": "aios", "api_base_url": "${base}/chat/completions", "api_key": "none", "models": ["${model}"] }
# then:  ccr code
# (Without a router, Claude Code cannot use an OpenAI-compatible endpoint.)`;
  else if(kind==='continue')t=`# Continue (~/.continue/config.yaml) - Cline/Roo: pick "OpenAI Compatible", same Base URL / model
models:
  - name: ${model}
    provider: openai
    apiBase: ${base}
    apiKey: none
    model: ${model}
    roles: [chat, edit, apply, autocomplete]`;
  else if(kind==='dsh')t=`# dsh (DeepSeek Harness): add a custom OpenAI-compatible provider
dsh provider add aios --type openai --base-url ${base} --api-key none --model ${model}
dsh --provider aios`;
  else if(kind==='openwebui')t=`# Open WebUI -> Admin Settings -> Connections -> OpenAI API
URL:  ${base}
Key:  none
# models appear automatically as "${model}"`;
  else t=`# any OpenAI SDK / tool
set OPENAI_BASE_URL=${base}
set OPENAI_API_KEY=none
# model name: ${model}`;
  $('#agent-snippet').textContent=t;
}
$('#agent-kind').onchange=renderAgent;$('#agent-addr').onchange=renderAgent;
$('#agent-copy').onclick=()=>{navigator.clipboard.writeText($('#agent-snippet').textContent);toast('Copied');};

/* ---------- benchmark ---------- */
let BENCH_PROMPTS=[], BENCH_RUNS=[];
function benchStages(){return{ngpu:$('#bs-ngpu').checked,spec:$('#bs-spec').checked,split:$('#bs-split').checked,fa:$('#bs-fa').checked,kv:$('#bs-kv').checked,ub:$('#bs-ub').checked,slots:$('#bs-slots').checked,offload:$('#bs-offload').checked,prompts:$('#bs-prompts').checked,judge:$('#bs-judge').checked};}
function fmtBest(b){if(!b)return '—';return `${b.gpus?b.gpus.length+' GPU · ':''}${b.sm||'single'} · fa ${b.fa?'on':'off'} · KV ${b.kv} · ub ${b.ub}${b.ncmoe?' · ncmoe '+b.ncmoe:''}${b.parallel>1?' · slots '+b.parallel:''}${b.spec&&b.spec!=='none'?' · spec '+b.spec:''}`;}
async function loadBench(){
  try{const j=await api('/api/bench/results');BENCH_PROMPTS=j.prompts;BENCH_RUNS=j.runs;}catch(e){return;}
  const prev=new Set([...document.querySelectorAll('#b-prompts input:checked')].map(c=>c.value));
  $('#b-prompts').innerHTML=BENCH_PROMPTS.map(p=>`<label class="chk" style="margin:0"><input type="checkbox" value="${esc(p.id)}" ${prev.size&&!prev.has(p.id)?'':'checked'}> ${esc(p.name)}</label>`).join('');
  const tb=$('#bench-runs tbody');tb.innerHTML='';
  for(const r of [...BENCH_RUNS].reverse()){const b=r.best||{};const agg=(r.slots||[]).reduce((m,s)=>Math.max(m,s.agg_tps||0),0);
    const tr=document.createElement('tr');tr.innerHTML=`<td>${esc(r.time)}</td><td>${esc(r.model)}</td><td>${esc((r.backend||'').toUpperCase())} · ${(r.gpus||[]).length}× ${esc((r.gpus||[''])[0]||'')}</td><td style="font-size:.8rem">${esc(fmtBest(b))}</td><td class="num">${b.tg??'—'}</td><td class="num">${b.pp??'—'}</td><td class="num">${agg||'—'}</td><td class="num">${r.avg_score??'—'}</td><td><button class="btn" data-id="${r.id}">Open</button></td>`;
    tb.appendChild(tr);}
  tb.querySelectorAll('button').forEach(b=>b.onclick=()=>showBench(b.dataset.id));
}
function fillBenchModels(){
  const main=LOCAL.filter(m=>!m.mmproj&&!m.draft&&(!m.shard||/00001-of/.test(m.name)));
  const cur=$('#b-model').value,curj=$('#b-judge').value;
  $('#b-model').innerHTML=main.map(m=>`<option value="${esc(m.path)}">${esc(m.name)} (${fmtGB(m.size_gb)})</option>`).join('')||'<option value="">— no models —</option>';
  $('#b-judge').innerHTML='<option value="self">same model</option><option value="">none</option>'+main.map(m=>`<option value="${esc(m.path)}">${esc(m.name)}</option>`).join('');
  if(cur)$('#b-model').value=cur;if(curj)$('#b-judge').value=curj;
  const curd=$('#b-draft').value;$('#b-draft').innerHTML='<option value="">none</option>'+LOCAL.filter(m=>m.draft).map(m=>`<option value="${esc(m.path)}">${esc(m.name)}</option>`).join('');if(curd)$('#b-draft').value=curd;
}
async function runBench(){
  const model=$('#b-model').value;if(!model)return toast('Download a model first');
  const prompts=[...document.querySelectorAll('#b-prompts input:checked')].map(c=>c.value);
  $('#log-bench').innerHTML='';$('#bench-out').innerHTML='';setPill('#pill-bench','running','busy');$('#btn-bench').disabled=true;$('#btn-bench-cancel').disabled=false;
  try{const {job}=await post('/api/bench/run',{model,draft:$('#b-draft').value,judge:$('#b-judge').value,stages:benchStages(),prompts,backend:HW.backend,gpus:selectedGpus(),repo:$('#repo').value.trim()});
    $('#btn-bench-cancel').onclick=()=>post('/api/jobs/cancel',{id:job.id});
    stream(`/api/jobs/${job.id}/stream`,$('#log-bench'),async()=>{const j=await api(`/api/jobs/${job.id}`);setPill('#pill-bench',j.status,j.status==='done'?'ok':'crit');$('#btn-bench').disabled=false;$('#btn-bench-cancel').disabled=true;await loadBench();if(j.result)showBench(j.result);pollServer();});
  }catch(e){toast(e.message);setPill('#pill-bench','failed','crit');$('#btn-bench').disabled=false;$('#btn-bench-cancel').disabled=true;}
}
async function showBench(id){
  const r=await api(`/api/bench/results?id=${id}`);const out=$('#bench-out');const b=r.best||{};
  let h=`<div class="note" style="margin-top:.8rem"><b>${esc(r.model)}</b> on ${esc((r.backend||'').toUpperCase())} with ${(r.gpus||[]).length} card(s) · ${esc(r.time)}${r.judge?` · judged by ${esc(r.judge)}`:''}</div>`;
  h+=`<div class="row" style="align-items:center;gap:1rem"><div><div class="k" style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em">Best settings</div><div><b>${esc(fmtBest(b))}</b> → pp512 <b>${b.pp??'—'}</b> t/s · tg128 <b>${b.tg??'—'}</b> t/s${b.agg_tps?` · ${b.agg_tps} t/s aggregate at ${b.parallel} slot${b.parallel>1?'s':''}`:''}</div></div><button class="btn primary" id="bench-apply">Use these settings</button></div>`;
  if(r.spec&&r.spec.length){h+=`<div class="tw"><table><thead><tr><th>Speculative</th><th class="num">tok/s</th><th class="num">TTFT s</th><th>Note</th></tr></thead><tbody>`+r.spec.map(x=>`<tr><td>${esc(x.spec)}${x.kind?' ('+esc(x.kind)+')':''}</td><td class="num">${x.tok_s??'—'}</td><td class="num">${x.ttft??'—'}</td><td style="color:var(--muted);font-size:.8rem">${esc(x.err||'')}</td></tr>`).join('')+`</tbody></table></div>`;}
  if(r.sweep&&r.sweep.length){h+=`<div class="tw"><table><thead><tr><th>GPUs</th><th>Split</th><th>FA</th><th>K / V</th><th>ub</th><th>ncmoe</th><th class="num">pp512 t/s</th><th class="num">tg128 t/s</th></tr></thead><tbody>`;
    const bestTg=Math.max(...r.sweep.map(x=>x.tg||0));
    for(const x of r.sweep)h+=`<tr class="${x.tg===bestTg?'best':''}"><td>${x.ngpu??'—'}</td><td>${esc(x.sm||'—')}</td><td>${x.fa?'on':'off'}</td><td class="mono">${esc(x.ctk)} / ${esc(x.ctv)}</td><td>${x.ub}</td><td>${x.ncmoe||'—'}</td><td class="num">${x.pp??'—'}</td><td class="num">${x.tg??'—'}</td></tr>`;
    h+=`</tbody></table></div>`;}
  if(r.slots&&r.slots.length){h+=`<div class="tw"><table><thead><tr><th>Slots</th><th class="num">Aggregate t/s</th><th class="num">Per request t/s</th><th class="num">TTFT s</th></tr></thead><tbody>`+r.slots.map(x=>`<tr><td>${x.slots}</td><td class="num">${x.agg_tps}</td><td class="num">${x.per_req_tps}</td><td class="num">${x.ttft}</td></tr>`).join('')+`</tbody></table></div>`;}
  if(r.prompts&&r.prompts.length){h+=`<h3 style="margin-top:1rem;font-size:.95rem">Prompt suite${r.avg_score!=null?` · average ${r.avg_score}/10`:''}</h3>`;
    for(const p of r.prompts){h+=`<div class="panel" style="margin-top:.6rem;padding:.8rem 1rem"><div style="display:flex;gap:.8rem;align-items:center;flex-wrap:wrap"><b>${esc(p.name)}</b>${p.score!=null?`<span class="pill ${p.score>=7?'ok':p.score>=4?'warn':'crit'}">${p.score}/10</span>`:''}<span style="color:var(--muted);font-size:.85rem">${p.tokens} tokens · TTFT ${p.ttft}s · ${p.tok_s} t/s</span><span class="spacer"></span>${p.html?`<button class="btn" data-prev="${esc(p.id)}">Preview</button>`:''}<button class="btn" data-src="${esc(p.id)}">Output</button></div>${p.reason?`<div style="font-size:.85rem;color:var(--ink-2);margin-top:.3rem">${esc(p.reason)}</div>`:''}<div id="bp-${esc(p.id)}" hidden style="margin-top:.6rem"></div></div>`;}}
  out.innerHTML=h;
  $('#bench-apply').onclick=()=>{$('#s-kv').value=b.kv||'q8_0';$('#s-split').value=b.sm||'layer';$('#s-ub').value=String(b.ub||'');$('#s-fa').value=b.fa?'on':'off';$('#s-ncmoe').value=b.ncmoe||0;$('#s-par').value=b.parallel||1;$('#s-spec').value=b.spec&&b.spec!=='none'?b.spec:'none';if(b.draft){$('#s-draft').value=b.draft;}if(b.gpus){const on=new Set(b.gpus);document.querySelectorAll('#s-gpus input').forEach(c=>{c.checked=on.has(Number(c.value));});}if(r.path){$('#model-file').value=r.path;}updateCmd();toast('Serve settings updated from the benchmark');$('#s-cmd').scrollIntoView({behavior:'smooth',block:'center'});};
  out.querySelectorAll('[data-prev]').forEach(btn=>btn.onclick=()=>{const p=r.prompts.find(x=>x.id===btn.dataset.prev);const box=$('#bp-'+p.id);box.hidden=!box.hidden;box.innerHTML=box.hidden?'':`<iframe sandbox="allow-scripts allow-pointer-lock" style="width:100%;height:520px;border:1px solid var(--line);border-radius:8px;background:#fff" srcdoc="${p.html.replace(/&/g,'&amp;').replace(/"/g,'&quot;')}"></iframe>`;});
  out.querySelectorAll('[data-src]').forEach(btn=>btn.onclick=()=>{const p=r.prompts.find(x=>x.id===btn.dataset.src);const box=$('#bp-'+p.id);box.hidden=!box.hidden;box.innerHTML=box.hidden?'':`<pre style="max-height:420px;overflow:auto;background:var(--code-bg);color:var(--code-ink);padding:.8rem;border-radius:8px;font-size:.78rem;white-space:pre-wrap">${esc(p.text)}</pre>`;});
  out.scrollIntoView({behavior:'smooth',block:'start'});
}
$('#btn-bench').onclick=runBench;
$('#bp-save').onclick=async()=>{try{await post('/api/bench/prompts',{prompt:{name:$('#bp-name').value,prompt:$('#bp-prompt').value,html:$('#bp-html').checked,max_tokens:$('#bp-max').value,rubric:$('#bp-rubric').value}});toast('Prompt saved');await loadBench();}catch(e){toast(e.message);}};

/* ---------- hardware ---------- */
async function loadHW(){
  setPill('#pill-hw','detecting','busy');
  const be=$('#backend').value;
  HW=await api('/api/hardware?fresh=1'+(be&&be!=='auto'?`&backend=${be}`:''));
  $('#root').textContent=HW.root;$('#ver').textContent='v'+HW.version;
  const c=HW.cpu,r=HW.ram,o=HW.os;
  $('#hw-kv').innerHTML=`
    <div class="k">host</div><div class="v">${esc(o.hostname)} · ${esc(o.system)} ${esc(o.release)}</div>
    <div class="k">cpu</div><div class="v">${esc(c.name||'?')}<br><span style="color:var(--muted);font-size:.8rem">${c.cores} cores / ${c.threads} threads${c.avx512?' · AVX-512':''}</span></div>
    <div class="k">ram</div><div class="v">${r.total_gb} GB <span style="color:var(--muted);font-size:.8rem">(${r.free_gb} free)</span></div>
    <div class="k">vram</div><div class="v">${HW.vram_total_gb} GB usable across ${HW.compute_gpus.length} ${HW.backend.toUpperCase()} GPU${HW.compute_gpus.length==1?'':'s'}</div>
    <div class="k">backend</div><div class="v">${HW.backend.toUpperCase()}${HW.backend!==HW.recommended_backend?` <span style="color:var(--muted);font-size:.8rem">(detected: ${HW.recommended_backend})</span>`:''}</div>`;
  $('#hw-gpus').innerHTML=HW.gpus.map(g=>`<div class="gpu" style="${g.compute?'':'opacity:.55'}"><div class="name">GPU${g.index} · ${esc(g.name)}${g.integrated?' <span class="tag">integrated</span>':''}${g.compute?'':' <span class="tag">not used by '+HW.backend+'</span>'}</div>
    <div class="meta">${g.vram_gb} GB${g.used_gb!=null?` (${g.used_gb} used)`:''}${g.compute_cap?` · cc ${g.compute_cap}`:''}${g.driver?` · driver ${esc(g.driver)}`:''}${g.pcie_width?` · PCIe x${g.pcie_width}`:''}</div></div>`).join('')
    ||'<div class="note warn">No GPU detected — builds will be CPU-only.</div>';
  const t=HW.toolchain;
  $('#hw-tools').innerHTML=Object.entries(t).map(([k,v])=>`<span class="${v.path?'ok':'miss'}" title="${esc(v.path||'not found')}">${k}${v.version?' '+esc(v.version):''}</span>`).join('');
  const missing=Object.entries(t).filter(([k,v])=>!v.path&&['git','cmake'].includes(k)).map(([k])=>k);
  let note=missing.length?`Missing ${missing.join(', ')} — the build script will install them. `:'';
  const sc=HW.scripts;
  if(!sc.present)note+=`${sc.name} is missing from ${HW.root}.`;
  else if(!sc.supports_backend)note+=`${sc.name} in ${HW.root} is v${sc.version} (need v${sc.expected}) — it will be rewritten automatically on the next Build.`;
  $('#hw-note').textContent=note;$('#hw-note').className='note'+(note?' warn':'');
  setPill('#pill-hw','detected','ok');
  $('#backend').options[0].text=`auto (${HW.recommended_backend})`;
  $('#build-script').textContent=o.system==='Windows'?'build-llama.ps1':'build-llama.sh';
  renderBuild(HW.build);
  renderGpuPicker();
}
function selectedGpus(){const cbs=[...document.querySelectorAll('#s-gpus input[type=checkbox]')];if(!cbs.length)return '';const on=cbs.filter(c=>c.checked).map(c=>c.value);return on.length===cbs.length?'':on.join(',');}
function renderGpuPicker(){
  const all=HW.all_compute_gpus||HW.compute_gpus||[];const prev=new Set((selectedGpus()||'').split(',').filter(Boolean).map(Number));
  $('#s-gpus').innerHTML=all.length?all.map(g=>`<label class="chk" style="margin:0"><input type="checkbox" value="${g.index}" ${prev.size&&!prev.has(g.index)?'':'checked'}> GPU${g.index} · ${g.vram_gb} GB</label>`).join(''):'<span style="color:var(--muted)">none for this backend</span>';
  document.querySelectorAll('#s-gpus input').forEach(c=>c.addEventListener('change',()=>{const n=[...document.querySelectorAll('#s-gpus input:checked')].length;$('#s-split').disabled=n<2;updateCmd();if(QUANTS)listQuants();}));
  $('#s-split').disabled=(all.length)<2;
}
function renderBuild(b){
  if(!b.present){$('#build-kv').innerHTML='<div class="k">status</div><div class="v">not built yet</div>';setPill('#pill-build','build: none','warn');return;}
  const i=b.info||{};
  const gpus=Array.isArray(i.gpus)?i.gpus.map(g=>g.Name||g.name).join(', '):'';
  $('#build-kv').innerHTML=`<div class="k">status</div><div class="v">ready</div>
    <div class="k">llama.cpp</div><div class="v mono">${esc(i.llama_tag||'?')}</div>
    <div class="k">backend</div><div class="v">${esc((i.backend||'cuda').toUpperCase())}${i.cuda?' · CUDA '+esc(i.cuda):''}${i.cuda_archs?' · sm '+esc(i.cuda_archs):''}${i.oneapi_root?' · oneAPI '+esc(i.oneapi_ver||''):''}</div>
    <div class="k">built</div><div class="v">${esc((i.built_at||'').replace('T',' ').slice(0,16))}</div>
    ${gpus?`<div class="k">for</div><div class="v" style="font-size:.8rem">${esc(gpus)}</div>`:''}`;
  setPill('#pill-build','build: '+(i.llama_tag||'ready'),'ok');
}

/* ---------- build ---------- */
async function build(){
  const el=$('#log-build');el.innerHTML='';
  $('#btn-build').disabled=true;$('#btn-build-cancel').hidden=false;setPill('#pill-buildjob','building','busy');
  try{
    const {job}=await post('/api/build',{backend:$('#backend').value,clean:$('#clean').checked,branch:branchValue()});
    $('#btn-build-cancel').dataset.id=job.id;
    stream(`/api/jobs/${job.id}/stream`,el,async()=>{
      const j=await api(`/api/jobs/${job.id}`);
      setPill('#pill-buildjob',j.status,j.status==='done'?'ok':'crit');
      $('#btn-build').disabled=false;$('#btn-build-cancel').hidden=true;
      HW=await api('/api/hardware');renderBuild(HW.build);
    });
  }catch(e){toast(e.message);$('#btn-build').disabled=false;$('#btn-build-cancel').hidden=true;setPill('#pill-buildjob','error','crit');}
}

/* ---------- models ---------- */
async function loadReleases(){
  try{
    const j=await api('/api/releases');const sel=$('#branch-sel');
    const cur=sel.value;
    sel.innerHTML='<option value="master">master (latest commit)</option>'+j.releases.map(r=>`<option value="${esc(r.tag)}">${esc(r.tag)} · ${esc(r.date)}${r.prerelease?' (pre)':''}</option>`).join('')+'<option value="__custom">custom…</option>';
    if(j.error){const o=document.createElement('option');o.disabled=true;o.textContent='(release list unavailable: '+j.error+')';sel.insertBefore(o,sel.lastElementChild);}
    if([...sel.options].some(o=>o.value===cur))sel.value=cur;
    $('#rel-link').href=j.releases_url;
  }catch(e){}
}
function branchValue(){const v=$('#branch-sel').value;return v==='__custom'?$('#branch').value.trim():v;}
async function loadCatalog(){
  const qs=HW?`?backend=${HW.backend}&gpus=${encodeURIComponent(selectedGpus())}&ctx=${$('#ctx').value}&kv=${$('#kv').value}&priority=${$('#prio').value}`:'';
  const j=await api('/api/catalog'+qs);CATALOG=j.catalog;const m=j.machine;
  $('#catalog-note').style.display='';$('#catalog-note').textContent=m?`Ranked for this machine: ${m.cards} ${m.backend.toUpperCase()} card${m.cards===1?'':'s'}, ${m.vram_total_gb} GB VRAM, ${m.ram_gb} GB RAM, ${$('#ctx').value} context, ${$('#prio').value} priority. Sizes are estimates until you open a model.`:'';
  $('#catalog').innerHTML=CATALOG.map(c=>{const v=c.verdict||{};const cls=v.fit==='fits'?'ok':v.fit==='offload'?'warn':'crit';
    return `<div class="card ${v.fit==='none'?'dim':''}" data-repo="${esc(c.repo)}"><div class="t">${esc(c.name)}${v.best?' <span class="pill ok" style="font-size:.65rem">best for this machine</span>':''}</div><div class="r">${esc(c.repo)}</div><div class="b">${esc(c.blurb)}</div>
    <div class="fit ${cls}">${v.tag?`<b>${esc(v.tag)}</b> ~${v.size_gb} GB · ${esc(v.mode)}`:`~${v.size_gb} GB even at 2-bit · ${esc(v.mode)}`}</div>
    <div class="tags"><span class="tag">${esc(c.arch)}</span>${c.mtp?'<span class="tag">MTP</span>':''}${c.draft_repo?'<span class="tag">MTP draft file</span>':''}${c.vision?'<span class="tag">vision</span>':''}<span class="tag">${c.kv_kb_per_token} KB/tok KV</span></div></div>`;}).join('');
  document.querySelectorAll('.card').forEach(el=>el.onclick=()=>{$('#repo').value=el.dataset.repo;listQuants();});
}
async function listQuants(){
  const repo=$('#repo').value.trim();if(!repo)return;
  document.querySelectorAll('.card').forEach(el=>el.classList.toggle('sel',el.dataset.repo===repo));
  $('#rec-note').textContent='Fetching file list from Hugging Face…';$('#quants').hidden=true;$('#budget').hidden=true;
  try{
    QUANTS=await api(`/api/hf/files?repo=${encodeURIComponent(repo)}&ctx=${$('#ctx').value}&kv=${$('#kv').value}&priority=${$('#prio').value}&backend=${HW.backend}&gpus=${encodeURIComponent(selectedGpus())}`);
  }catch(e){$('#rec-note').textContent=e.message;return;}
  SEL=QUANTS.entry;const R=QUANTS.recommendation;
  const tb=$('#quants tbody');tb.innerHTML='';
  for(const q of QUANTS.quants){
    const fits=R.fits.includes(q.tag),best=q.tag===R.best,cpu=q.tag===R.cpu_fallback;
    const tr=document.createElement('tr');tr.className=best?'best':fits?'':'nofit';
    tr.innerHTML=`<td class="mono">${esc(q.tag)}</td><td class="num">${fmtGB(q.size_gb)}</td><td>${q.files.length}${q.sharded?' shards':''}</td>
      <td>${best?'★ recommended':fits?'fits':(R.offload||[]).includes(q.tag)?'fits with experts in RAM (--n-cpu-moe)':cpu?'RAM offload only':'too big'}</td>
      <td><button class="btn" data-tag="${esc(q.tag)}">${LOCAL.some(m=>q.files.some(f=>f.endsWith(m.name)))?'Re-download':'Download'}</button></td>`;
    tb.appendChild(tr);
  }
  tb.querySelectorAll('button').forEach(b=>b.onclick=()=>download(repo,b.dataset.tag));
  $('#quants').hidden=false;
  const mm=QUANTS.mmproj.length;$('#want-mmproj').disabled=!mm;$('#want-mmproj').parentElement.style.opacity=mm?1:.5;
  const hasDraft=(QUANTS.drafts||[]).length>0||!!(QUANTS.entry&&QUANTS.entry.draft_repo);$('#want-draft').disabled=!hasDraft;$('#want-draft').parentElement.style.opacity=hasDraft?1:.5;
  const bestQ=QUANTS.quants.find(q=>q.tag===R.best);
  if(bestQ){
    $('#rec-note').innerHTML=`Recommended <b>${esc(R.best)}</b> (${fmtGB(bestQ.size_gb)}): the ${R.priority==='speed'?'fastest good-quality quant':'highest quality'} that fits ${HW.vram_total_gb} GB of VRAM with ${R.kv_gb} GB of KV cache at ${$('#ctx').value} tokens and ~${R.overhead_gb} GB of buffers.`;
    renderBudget(bestQ.size_gb,R.kv_gb,R.overhead_gb,HW.vram_total_gb);
  }else if(R.cpu_fallback){
    $('#rec-note').innerHTML=`Nothing fits entirely in ${HW.vram_total_gb} GB of VRAM. <b>${esc(R.cpu_fallback)}</b> can run with some layers in system RAM (set GPU layers below 999) — expect a big speed drop.`;
  }else if(!QUANTS.quants.length){$('#rec-note').textContent='No GGUF files found in that repo — check the name (it usually ends in -GGUF).';
  }else{$('#rec-note').textContent='No quant in this repo fits this machine.';}
}
function renderBudget(w,k,o,total){
  const free=Math.max(0,total-w-k-o);
  $('#bar').innerHTML=`<span style="flex:${w};background:var(--bar)">weights ${w} GB</span><span style="flex:${k};background:var(--bar2)">KV ${k} GB</span><span style="flex:${o};background:var(--bar3)">${o} GB</span><span class="free" style="flex:${free};background:var(--bar-free)">free ${free.toFixed(1)} GB</span>`;
  $('#budget').hidden=false;
}
async function download(repo,tag){
  const el=$('#log-dl');el.innerHTML='';$('#dl-progress').hidden=false;setPill('#pill-dl','downloading','busy');
  try{
    const {job}=await post('/api/download',{repo,quant:tag,mmproj:$('#want-mmproj').checked,draft:$('#want-draft').checked});
    const tick=setInterval(async()=>{try{const j=await api(`/api/jobs/${job.id}`);if(j.progress!=null)$('#dl-progress i').style.width=(j.progress*100).toFixed(1)+'%';}catch{}},1500);
    stream(`/api/jobs/${job.id}/stream`,el,async()=>{clearInterval(tick);const j=await api(`/api/jobs/${job.id}`);
      setPill('#pill-dl',j.status,j.status==='done'?'ok':'crit');await loadLocal();if(j.status==='done')toast('Model ready in models/');});
  }catch(e){toast(e.message);setPill('#pill-dl','error','crit');}
}
async function loadLocal(){
  const j=await api('/api/models/local');LOCAL=j.models;
  if(document.activeElement!==$('#models-dir'))$('#models-dir').value=j.dir;
  $('#models-free').textContent=(j.free_gb!=null?`${fmtGB(j.free_gb)} free · `:'')+`${LOCAL.length} GGUF file${LOCAL.length===1?'':'s'}`;
  const main=LOCAL.filter(m=>!m.mmproj&&!m.draft&&(!m.shard||/00001-of/.test(m.name)));
  const cur=$('#model-file').value;
  const curd=$('#s-draft').value;$('#s-draft').innerHTML='<option value="">none</option>'+LOCAL.filter(m=>m.draft).map(m=>`<option value="${esc(m.path)}">${esc(m.name)}</option>`).join('');if(curd)$('#s-draft').value=curd;
  $('#model-file').innerHTML=main.map(m=>`<option value="${esc(m.path)}">${esc(m.name)} (${fmtGB(m.size_gb)})</option>`).join('')||'<option value="">— nothing downloaded yet —</option>';
  if(cur)$('#model-file').value=cur;
  $('#mmproj-file').innerHTML='<option value="">none</option>'+LOCAL.filter(m=>m.mmproj).map(m=>`<option value="${esc(m.path)}">${esc(m.name)}</option>`).join('');
  fillBenchModels();
  updateCmd();
}

/* ---------- server ---------- */
function serveOpts(){return{backend:HW?HW.backend:'',gpus:selectedGpus(),model:$('#model-file').value,mmproj:$('#mmproj-file').value,draft:$('#s-draft').value,ctx:$('#s-ctx').value,kv:$('#s-kv').value,split:$('#s-split').value,ngl:$('#s-ngl').value,parallel:$('#s-par').value,mtp:$('#s-mtp').checked,ub:$('#s-ub').value,fa:$('#s-fa').value,ncmoe:$('#s-ncmoe').value,spec:$('#s-spec').value,port:$('#s-port').value,host:$('#s-host').value,extra:$('#s-extra').value,repo:$('#repo').value.trim()};}
let cmdTimer;
function updateCmd(){clearTimeout(cmdTimer);cmdTimer=setTimeout(async()=>{const o=serveOpts();if(!o.model){$('#s-cmd').textContent='';return;}
  const q=new URLSearchParams({...o,mtp:o.mtp?'1':'0'});try{const j=await api('/api/server/args?'+q);const envs=Object.entries(j.env||{}).map(([k,v])=>`${k}=${v}`).join(' ');$('#s-cmd').textContent=(envs?envs+' ':'')+'llama-server '+j.args.map(a=>/\s/.test(a)?JSON.stringify(a):a).join(' ');}catch{}},250);}
async function startServer(){
  const o=serveOpts();if(!o.model)return toast('Download a model first');
  $('#log-srv').innerHTML='';
  try{await post('/api/server/start',o);}catch(e){return toast(e.message);}
  stream('/api/server/stream',$('#log-srv'),()=>{});
}
let wasRunning=false;
async function pollServer(){
  try{const s=await api('/api/server/status');
    const cls=s.state==='ready'?'ok':s.running?'busy warn':'';
    setPill('#pill-srv',s.state,cls);setPill('#pill-server','server: '+s.state,cls);
    $('#btn-start').disabled=s.running;$('#btn-stop').disabled=!s.running;
    $('#endpoint').hidden=!s.running;
    if(s.running){const h=location.hostname;$('#ep-ui').href=`http://${h}:${s.port}/`;$('#ep-metrics').href=`http://${h}:${s.port}/metrics`;if(!wasRunning||!NET)await loadNet();}
    if(!s.running&&s.exit!=null&&s.exit!==0){setPill('#pill-srv','exited '+s.exit,'crit');if(wasRunning)toast(`llama-server exited with code ${s.exit} — see the log`);}
    wasRunning=s.running;
  }catch{}
}
async function testPrompt(){
  $('#answer').hidden=false;$('#answer').textContent='…';
  try{const j=await post('/api/server/chat',{prompt:$('#prompt').value});
    const t=j.timings||{};const tg=t.predicted_per_second?`${t.predicted_per_second.toFixed(1)} tok/s generation · ${(t.prompt_per_second||0).toFixed(0)} tok/s prompt`:`${j.tok_s} tok/s`;
    $('#answer').textContent=j.text.trim()+`\n\n— ${j.usage.completion_tokens} tokens in ${j.seconds}s · ${tg}`;
  }catch(e){$('#answer').textContent='Error: '+e.message;}
}

/* ---------- wiring ---------- */
$('#btn-refresh').onclick=loadHW;
$('#branch-sel').onchange=()=>{const c=$('#branch-sel').value==='__custom';$('#branch').hidden=!c;if(c)$('#branch').focus();};
$('#backend').onchange=async()=>{await loadHW();loadCatalog();if(QUANTS)listQuants();updateCmd();};
['#ctx','#kv','#prio'].forEach(sel=>$(sel).addEventListener('change',()=>{loadCatalog();}));
$('#btn-build').onclick=build;
$('#btn-build-cancel').onclick=()=>post('/api/jobs/cancel',{id:$('#btn-build-cancel').dataset.id});
$('#btn-repo').onclick=listQuants;$('#repo').addEventListener('keydown',e=>{if(e.key==='Enter')listQuants();});
$('#ctx').onchange=$('#kv').onchange=$('#prio').onchange=()=>{if(QUANTS)listQuants();$('#s-ctx').value=$('#ctx').value;$('#s-kv').value=$('#kv').value;updateCmd();};
['#model-file','#mmproj-file','#s-draft','#s-ctx','#s-kv','#s-split','#s-ngl','#s-par','#s-mtp','#s-ub','#s-fa','#s-ncmoe','#s-spec','#s-port','#s-host','#s-extra'].forEach(s=>{$(s).addEventListener('input',updateCmd);$(s).addEventListener('change',updateCmd);});
$('#btn-models-dir').onclick=async()=>{try{await post('/api/models/dir',{dir:$('#models-dir').value});await loadLocal();toast('Models folder set');}catch(e){toast(e.message);}};
$('#models-dir').addEventListener('keydown',e=>{if(e.key==='Enter')$('#btn-models-dir').click();});
$('#btn-start').onclick=startServer;$('#btn-stop').onclick=()=>post('/api/server/stop');$('#btn-test').onclick=testPrompt;

(async()=>{
  await loadHW();loadReleases();await loadCatalog();await loadLocal();
  // resume streams for anything already running
  try{const {jobs}=await api('/api/jobs');for(const j of jobs.filter(j=>j.status==='running')){
    const el=j.kind==='build'?$('#log-build'):$('#log-dl');stream(`/api/jobs/${j.id}/stream`,el,()=>{});}}catch{}
  const s=await api('/api/server/status');if(s.running)stream('/api/server/stream',$('#log-srv'),()=>{});
  try{const st=await api('/api/state');const l=st.last_server;if(l){for(const [k,id] of Object.entries({ctx:'#s-ctx',kv:'#s-kv',split:'#s-split',ngl:'#s-ngl',parallel:'#s-par',ub:'#s-ub',fa:'#s-fa',ncmoe:'#s-ncmoe',spec:'#s-spec',draft:'#s-draft',port:'#s-port',host:'#s-host',extra:'#s-extra'}))if(l[k]!=null)$(id).value=l[k];if(l.mtp!=null)$('#s-mtp').checked=!!l.mtp;if(l.gpus){const on=new Set(l.gpus.split(',').map(Number));document.querySelectorAll('#s-gpus input').forEach(c=>{c.checked=on.has(Number(c.value));});}if(l.model)$('#model-file').value=l.model;if(l.repo)$('#repo').value=l.repo;updateCmd();}}catch{}
  pollServer();setInterval(pollServer,3000);
  loadBench();
  // resume a benchmark that is still running
  try{const {jobs}=await api('/api/jobs');const b=jobs.find(j=>j.kind==='bench'&&j.status==='running');if(b){setPill('#pill-bench','running','busy');$('#btn-bench').disabled=true;$('#btn-bench-cancel').disabled=false;$('#btn-bench-cancel').onclick=()=>post('/api/jobs/cancel',{id:b.id});stream(`/api/jobs/${b.id}/stream`,$('#log-bench'),async()=>{const j=await api(`/api/jobs/${b.id}`);setPill('#pill-bench',j.status,j.status==='done'?'ok':'crit');$('#btn-bench').disabled=false;$('#btn-bench-cancel').disabled=true;await loadBench();if(j.result)showBench(j.result);});}}catch{}
  if(!$('#repo').value){$('#repo').value=CATALOG[0].repo;}listQuants();
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="AIOS - browser front end for building and running llama.cpp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7800)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--models-dir", default="", help="folder holding the GGUF files (default: <root>\\models, or AIOS_MODELS; a UNC/NAS path or another disk works)")
    a = ap.parse_args()
    global MODEL_DIR
    if a.models_dir:
        MODEL_DIR = a.models_dir
    elif not os.environ.get("AIOS_MODELS") and load_state().get("models_dir"):
        MODEL_DIR = load_state()["models_dir"]
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensure_scripts()
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    httpd.daemon_threads = True
    httpd.bind_host = a.host
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
    print(f"AIOS {APP_VERSION}  ->  {url}   (root: {ROOT})")
    if not a.no_browser:
        try:
            import webbrowser
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SERVER.stop()


if __name__ == "__main__":
    main()
