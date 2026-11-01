# llama.cpp build + Qwen3.8-27B tuning

Scripts that build llama.cpp from source for whatever CPU / RAM / GPU(s) they find,
and launch Qwen3.8-27B with settings tuned for the box they run on.
Default profile = **max decode speed on 2x RTX 5060 Ti 16 GB, Ryzen 9800X3D, 64 GB RAM**.

| File | OS | What it does |
|------|----|--------------|
| `build-llama.ps1` | Windows 11 | detect HW -> winget-install Git/CMake/Ninja/VS Build Tools/CUDA -> clone -> build -> `bin\` |
| `build-llama.sh`  | Ubuntu 22.04+ | detect HW -> apt-install toolchain + CUDA repo -> clone -> build -> `bin/` |
| `aios.py` | both | **Browser front end (v0.2.0)** — `python aios.py` opens http://localhost:7800: detects hardware, builds llama.cpp (via the scripts below) with live log, recommends + downloads a Hugging Face quant that fits your VRAM, starts/stops llama-server and lets you test it. No pip installs. **Self-contained:** it carries the four scripts below inside it and (re)writes them next to itself whenever they are missing or out of date, so `aios.py` is the only file you must copy. |
| `run-qwen.ps1` / `run-qwen.sh` | both | pick the right GGUF quant for the detected VRAM and start `llama-server -hf unsloth/Qwen3.8-27B-GGUF:<quant>` (llama.cpp downloads and caches it in `models/`) |

```powershell
# Easiest: the browser UI (Windows or Ubuntu, Python 3.9+)
python aios.py                 # opens http://localhost:7800 ; add --host 0.0.0.0 to reach it from the LAN
```
```powershell
# Windows (normal PowerShell, no admin needed unless winget has to install something)
powershell -ExecutionPolicy Bypass -File .\build-llama.ps1
.\run-qwen.ps1                       # server on http://0.0.0.0:8080
.\run-qwen.ps1 -Bench                # llama-bench numbers
.\run-qwen.ps1 -Split row -Bench     # compare row split
```
```bash
# Ubuntu
chmod +x *.sh && ./build-llama.sh && ./run-qwen.sh
```
If you ever see "A parameter cannot be found that matches parameter name Backend", an old script copy is being used: restart `aios.py` (a running Python process keeps old code in memory) and it rewrites the scripts.
Build scripts take `-Backend cuda|vulkan|sycl|cpu` / `--backend` (auto-detected by default).
* **cuda** - NVIDIA. **vulkan** - any discrete GPU (Intel Arc / AMD), needs only the Vulkan SDK.
* **sycl** - Intel's native oneAPI backend for Arc (A- and B-series incl. Arc Pro B60), per
  [llama.cpp docs/backend/SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md):
  installs the Intel oneAPI Base Toolkit (`winget install Intel.OneAPI.BaseToolkit` / Intel apt repo, several GB),
  builds with `cl` + `icx` (`icx` + `icpx` on Linux), `GGML_SYCL_F16=ON`, shared libs. SYCL binaries need the oneAPI
  runtime at run time: `aios.py` imports `setvars.bat|sh` into llama-server's environment automatically; by hand run
  `"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"` / `source /opt/intel/oneapi/setvars.sh` first.
  Auto picks SYCL only when an Arc card *and* oneAPI are already present, otherwise Vulkan.
  **Arc B-series needs DPC++ >= 2025.2**: llama.cpp's SYCL kernels reference the `intel_gpu_bmg_g31` architecture enum,
  which 2025.1 doesn't know (`no member named 'intel_gpu_bmg_g31'`). winget only offers 2025.1, so the Windows script
  downloads Intel's *Deep Learning Essentials* offline installer (2026.1, ~2 GB, UAC prompt) and installs it side by side.
* GPU detection reads the real VRAM from the display-class registry (`HardwareInformation.qwMemorySize`) - WMI's
  `AdapterRAM` is 32-bit and shows a 24 GB B60 as 4 GB - and marks integrated GPUs (`Intel(R) Graphics`, UHD/Iris,
  `Radeon(TM) Graphics`) so they are not counted as compute devices.
* winget installs use `--source winget`; the msstore source fails with `0x8a15005e` (certificate mismatch)
  on some networks and previously aborted every install.
Re-running the build script pulls the latest master and rebuilds. `-Clean` / `--clean` wipes the build dir.

## What the build does and why

* **`CMAKE_CUDA_ARCHITECTURES` from `nvidia-smi compute_cap`** - RTX 5060 Ti is compute 12.0, so `120`.
  sm_120 is *not* in llama.cpp's default arch list, and needs **CUDA Toolkit >= 12.8**; the scripts
  check the installed toolkit version and install a newer one if it is too old.
* **Ninja generator + nvcc** instead of the Visual Studio generator on Windows - avoids the
  "No CUDA toolset found" MSBuild-integration problem with Build Tools installs.
* **`BUILD_SHARED_LIBS=OFF`** - static binaries. On Linux this removes the trap where a rebuilt
  `llama-server` silently loads an old `libggml-cuda.so` and produces garbage output.
* **`GGML_NATIVE=ON`** - `-march=native` for the 9800X3D (Zen 5, AVX-512). Only matters for the
  little CPU-side work; don't copy the binaries to a different CPU.
* **HTTPS built in** - `LLAMA_BUILD_BORINGSSL=ON` on Windows, `LLAMA_OPENSSL=ON` + `libssl-dev` on Ubuntu, so `-hf repo:quant`
  downloads work. llama.cpp no longer uses libcurl; without an SSL library `-hf` fails with "HTTPS support disabled".
* **`GGML_CUDA_FA_ALL_QUANTS=ON`** - compiles flash-attention kernels for quantized KV cache
  (`q8_0` / `q4_0`) so `-fa on` + `--cache-type-k q8_0` actually runs on the GPU.
* **Full clone, never `--depth 1`** - shallow clones report "Already up to date" while sitting
  hundreds of commits behind; a stale build is the #1 cause of corrupted Qwen3.8 output
  (the DeltaNet linear-attention CUDA path was fixed on master).
* Ubuntu: if the system GCC is newer than the CUDA toolkit supports, an older `g++-N` is installed
  and passed as `CMAKE_CUDA_HOST_COMPILER`.

## Why these Qwen3.8-27B settings (speed profile, 2x16 GB)

Qwen3.8-27B is a hybrid model: 48 linear-attention (DeltaNet) layers + 16 full-attention layers.
Only the 16 full layers hold KV cache, so context is cheap: about **64 KB/token at f16**
(8 GB at 128K, 0.5 GB at 8K).

| Quant | Size | Fits 2x16 GB? |
|------:|-----:|:--------------|
| Q3_K_M | 13.8 GB | one card |
| IQ4_XS | 15.7 GB | one card, but no room for KV + MTP + compute buffers |
| **UD-Q4_K_XL** | **~17.5 GB** | **split, ~12 GB left for context** (default; Unsloth dynamic, better than Q4_K_M at the same size) |
| Q4_K_M | 17.1 GB | split, ~13 GB left |
| Q5_K_M | 19.8 GB | split, ~10 GB left |
| Q6_K | 22.9 GB | split, ~7 GB left |
| Q8_0 | 29.0 GB | split, only tiny context |

* **UD-Q4_K_XL split across both cards (`-sm layer`)** - the model does not fit one 16 GB card, so it
  is split. Layer split runs the two cards sequentially, so decode speed is roughly a single
  5060 Ti's 448 GB/s of bandwidth (~40 tok/s class before MTP). `-sm row` can pull bandwidth from
  both cards but is often slower over PCIe without NVLink - run `-Bench` both ways and keep the winner.
* **MTP speculative decoding** (`--spec-type draft-mtp --spec-draft-n-max 2 --parallel 1`) - the
  GGUF already contains the multi-token-prediction head; enabling it gives **+33-39 % decode speed**
  with acceptance ~0.8. It currently requires `--parallel 1`; pass `-NoMtp` / `--no-mtp` if you
  need multiple concurrent slots.
* **`--jinja` is mandatory** - without the proper chat template the model either rambles past its
  stop token or gets clipped.
* **`-c 65536` with `q8_0` KV** - ~4 GB of cache, comfortably inside the headroom. For the full
  256K context use `--ctx 262144 --kv q4_0 --no-mtp`.
* **`-fa on`** - flash attention; needed for quantized KV and for the MTP path.
* **`-b 2048 -ub 512`** - prompt-processing batch; larger `-ub` costs VRAM for little gain here.
* **`-t 8`** - physical cores of the 9800X3D; nearly irrelevant once everything is offloaded.
* **`--mmproj`** (`-Vision`) - adds the 0.9 GB vision encoder for image input.
* **`--metrics`** - Prometheus endpoint at `/metrics`, handy for the latency monitor.

## Machine profile: 2x Arc Pro B60 on ASUS ProArt X670E-Creator (Ryzen 9 9950X3D, 64 GB, Windows 11)

* Cards in **PCIEX16_1 and PCIEX16_2** (CPU lanes, x8/x8 - the B60 is x8 electrical, so each card gets its full
  Gen5 x8). Never a GPU in PCIEX16_3 (chipset, x2, disabled by default; enabling it halves M.2_3).
* Backend: **SYCL** (auto-picked once oneAPI >= 2025.2 is installed). Vulkan also works; AIOS adds
  `VK_LOADER_DRIVERS_DISABLE=*radeon*` on Vulkan so the Ryzen iGPU's driver stays out of the loader.
* ~41.6 GB usable VRAM (20.8 GB per card as the driver exposes it); weights budget ~37 GB at 32K context.
  Picks: Qwen3.8-27B Q8_0 (~29 GB, MTP), Qwen3.6-35B-A3B Q4_K_M (~21 GB), Qwen3-Coder-30B-A3B UD-Q4_K_XL (~18 GB),
  dense 32B at Q5/Q6. 70B only at 3-bit - skip.
* Split: `layer` for MoE; for dense models benchmark `tensor` (SYCL tensor parallelism, tuned for exactly 2 cards)
  against `layer`. Or one model per card via the GPU picker (`ONEAPI_DEVICE_SELECTOR=level_zero:N`), separate ports.
* 64 GB RAM > VRAM: every model that fits the cards also fits RAM, so AIOS uses `--load-mode none` (read in, no mmap).
* BIOS: Advanced -> PCI Subsystem Settings -> **Above 4G Decoding = Enabled**, **Re-Size BAR Support = Enabled**;
  Boot -> CSM disabled. A BIOS update resets these and a card "vanishes" / shows Code 12. If a card trains
  unreliably at Gen5, set the slot Link Speed to Gen4 - no loss at x8.
* 9950X3D = Zen 5: `GGML_NATIVE` gives AVX-512. The AMD iGPU is marked integrated and ignored for compute.
* MoE on SYCL: `-fa off` (V cache f16) and `GGML_SYCL_DISABLE_OPT=1`; KV `q4_0` is fastest on Battlemage, f16 above ~16K on SYCL.
* The previous 4x B60 / ROMED8-2T profile lives on in `runbook-4xb60.html` for reference.

## Machine profile: 5x RTX 5060 Ti 16 GB on ASRock Rack ROMED8-2T (EPYC 7443, 128 GB, Windows 11)

* Backend CUDA (sm_120, toolkit >= 12.8 - the build script installs it). 80 GB VRAM across five Gen4 x16 slots.
* **Qwen3.6-35B-A3B** (`unsloth/Qwen3.6-35B-A3B-GGUF`): UD-Q4_K_XL 22 GB / UD-Q6_K_XL 32 GB / Q8_0 37 GB. KV is tiny
  (hybrid DeltaNet, 2 KV heads), so run 128K+ context. With 3B active the benchmark's **GPU count** stage usually finds
  2-3 cards faster than all five (fewer PCIe hops) - the rest stay free for a second model. MTP draft is a separate
  GGUF (`...-MTP-GGUF`, tick "also download MTP / draft GGUF"); the speculative stage tests it as `draft-mtp` and `draft`.
* **Qwen3.8-Flash-Next** (`unsloth/Qwen3.8-Flash-Next-GGUF`): 125B / 6B active, UD-Q4_K_XL 111 GB, UD-IQ4_XS 94 GB,
  UD-Q3_K_XL 90 GB. Nothing fits 80 GB of VRAM: the benchmark finds the smallest `--n-cpu-moe` that loads (experts in
  the 128 GB of RAM); expect Q4_K_XL to need ~40 GB of experts in RAM and the 51B n-gram embedding to stay CPU-side.
  The MTP draft is in the same repo. Keep `--load-mode auto` (mmap) for this one - AIOS does when experts are offloaded.
* Expected sweep per model: GPU count -> split (layer vs row) -> flash-attn -> KV -> micro-batch -> slots 1/4/8 ->
  speculative (none / draft-mtp / draft / n-gram) -> prompt suite -> judge. Sampling for the prompt suite: thinking
  mode temp 1.0 / top_p 0.95 / top_k 20; coding temp 0.6 (Extra flags).
* BIOS: Above 4G Decoding + Re-Size BAR on, CSM off; five 24 GB BARs need it just like Arc. PSU: 5 x 180 W + CPU ~ 1.2 kW.

## Benchmark (section 4, v0.4.0+)
* Pick a local GGUF; the tool sweeps settings with `bin\llama-bench` on the detected cards (all of them in parallel when
  there is more than one): split mode (layer vs tensor on SYCL / row on CUDA), flash attention, KV cache type,
  micro-batch, then parallel slots (1/4/8 concurrent requests against a temporary llama-server on port 8089).
  MoE models bigger than VRAM are tried with experts in RAM (`--n-cpu-moe`, smallest value that loads).
* Then it runs the streamer prompt suite - Browser OS, Breakout, voxel world, hexagon balls, pelican SVG, to-do app,
  plus your own prompts - records TTFT / tokens/s, renders HTML results in a sandboxed iframe, and lets a judge model
  (same model or any other local GGUF) score each 1-10 against a rubric.
* Results live in `bench-results.json`; "Use these settings" fills section 3. The running llama-server is stopped
  during the slot test and the prompt suite.
* Serve section gained Micro-batch (`-ub`), Flash attn (`-fa`), Experts on CPU (`--n-cpu-moe`) and Speculative
  (MTP / n-gram / none). MoE models get `--cache-reuse 256 --defrag-thold 0.1` and `-ub 1024` by default; the
  recommender marks MoE quants that fit with experts in RAM and uses a 0.90 VRAM factor on SYCL.

## Models folder and downloads
* **Models folder** (section 2 of the GUI, `--models-dir`, or `AIOS_MODELS`): point AIOS at another disk, an existing
  LM Studio / llama.cpp model folder or a NAS share that already has the GGUFs; every `.gguf` under it is listed and
  downloads you already have are skipped. Persisted in `aios-state.json`.
* Downloads use 8 parallel HTTP ranges (`AIOS_DL_CONNECTIONS`), resumable (`.part` + `.parts.json`). A single
  Hugging Face stream is capped around 20-60 MB/s; eight usually fill a gigabit link (~110 MB/s). If it is still slow:
  Defender real-time scanning on the models folder (exclude it), Wi-Fi, or a HF CDN edge - `AIOS_HF_BASE` can point
  at a mirror.

## If something goes wrong

* `nvcc fatal: Unsupported gpu architecture 'compute_120'` -> CUDA toolkit < 12.8. Re-run; the script
  installs a newer one. On Windows uninstall the old toolkit first if `cudart64_12.dll` conflicts.
* `unsupported GNU version` (Ubuntu) -> handled automatically; or `--skip-install` and install `g++-14` yourself.
* Garbage / repeated tokens from the model -> old build. `git -C llama.cpp log -1`, re-run the build script,
  and make sure you are launching `bin/llama-server` and not a stale copy on `PATH`.
* `-hf` fails / "HTTPS support disabled" -> the build has no SSL library. Re-run the build script (it now enables it).
* winget `0x8a15005e: The server certificate did not match` -> the msstore source is blocked (TLS inspection).
  The scripts now pass `--source winget`; if it still fails, `winget source remove msstore`.
* `vswhere.exe is not recognized` -> VS Build Tools never got installed (see above); re-run, or install
  "Desktop development with C++" from visualstudio.microsoft.com/visual-cpp-build-tools.
* SYCL: `sycl-ls` shows no `[level_zero:gpu]` -> install/refresh the Intel Arc driver (Linux: add yourself to
  the `render` and `video` groups). `llama-server` fails to start with a missing `sycl*.dll`/`libsycl.so` ->
  run it through AIOS or after `setvars`. Allocation errors > 4 GiB -> `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` (AIOS sets it).
* `-hf` says no such file -> the quant tag must match a file in https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main;
  pass it with `-Quant` / `--quant` (e.g. `UD-Q4_K_XL`, `Q5_K_M`).

Sources: llama.cpp `docs/build.md`; ggml-org/llama.cpp discussion #27164 (Qwen3.8 corrupted output = stale build);
issue #22696 (Windows Blackwell sm_120); sudoingX/qwen38-mtp (MTP flags + benchmarks);
dev.to "Run Qwen 3.8 27B Locally" (quant sizes, KV maths, template).
