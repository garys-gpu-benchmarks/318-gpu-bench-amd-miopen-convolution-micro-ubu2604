# MIOpen Convolution Solver Microbenchmark Benchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/garys-gpu-benchmarks/318-gpu-bench-amd-miopen-convolution-micro-ubu2604/actions/workflows/ci.yml/badge.svg)](https://github.com/garys-gpu-benchmarks/318-gpu-bench-amd-miopen-convolution-micro-ubu2604/actions/workflows/ci.yml)

Target: Ubuntu 26.04 · AMD · see Hardware Requirements. This is a host benchmark, not a laptop `pip install` project.

## Quick Start

```bash
git clone https://github.com/garys-gpu-benchmarks/318-gpu-bench-amd-miopen-convolution-micro-ubu2604.git
cd 318-gpu-bench-amd-miopen-convolution-micro-ubu2604
sudo bash setup.sh --assume-yes
bash run_benchmark.sh --profile smoke --validate
```
Results are written to `results/benchmark.db` and `results/summary.json`.

This workload is executed on the validation host after the repository is copied there. `setup.sh` and `run_benchmark.sh` do not open an outbound SSH session.

Prerequisites: Ubuntu 26.04; AMD; Python 3.14.4; root or sudo for `setup.sh`. Framework: Bash, SQLite, Python, PyYAML, ROCm Runtime, HIP/ROCm, MIOpen, PyTorch-ROCm. This is a host benchmark, not a laptop `pip install` project.

```mermaid
flowchart LR
  setup.sh --> run_benchmark.sh --> parse_results.py --> results/benchmark.db
```

## 1. Overview

Runs MIOpenDriver convfp16 when present; otherwise falls back to torch.nn.functional.conv2d fp16. batch_size, input channels/height/width, output_channels, kernel_height, num_iterations, and conv_direction are used. warmup_iters applies only to the PyTorch fallback. conv_direction=all runs -F 1,2,4; otherwise -F 1. Sweep dimensions: dtype, input_source, conv_direction, batch_size, input_channels, input_height, input_width, output_channels.

## 2. What It Validates

- Validates MIOpenDriver convfp16 parse, or the PyTorch fp16 fallback row if the driver is missing
- #1: Convolution kernel execution time, forward (kernel_time_ms); is present and physically sensible.
- #2: Achieved compute throughput, forward (GFLOPs); is present and physically sensible.
- #3: Memory bandwidth, GB/s, forward (memory_bandwidth_gb_s); is present and physically sensible.
- #4: FLOP count (flopCnt) is present and physically sensible.

## 3. Metrics Captured

- **#1: Convolution kernel execution time, forward** — stored as `kernel_time_ms`.
- **#2: Achieved compute throughput, forward** — stored as `GFLOPs`.
- **#3: Memory bandwidth, GB/s, forward** — stored as `memory_bandwidth_gb_s`.
- **#4: FLOP count** — stored as `flopCnt`.

## 4. Hardware Requirements

### Supported environment

- OS: Ubuntu 26.04
- GPU vendor: AMD
- Framework family: Bash, SQLite, Python, PyYAML, ROCm Runtime, HIP/ROCm, MIOpen, PyTorch-ROCm
- Python: Python 3.14.4

### Reference validation environment

The tables below describe the machine used to generate the reference results. They are not a requirement that every user buy that exact cloud instance.

### System

Ubuntu 26.04 / AMD / Bash, SQLite, Python, PyYAML, ROCm Runtime, HIP/ROCm, MIOpen, PyTorch-ROCm

### GPU

Ubuntu 26.04 / AMD / Bash, SQLite, Python, PyYAML, ROCm Runtime, HIP/ROCm, MIOpen, PyTorch-ROCm

## 5. Software Requirements

| Component | Version |
|---|---|
| OS | Ubuntu 26.04 |
| Kernel | kernel 7.0.0 |
| Python | Python 3.14.4 |
| ROCm | ROCm 7.14 |
| rocBLAS | N/A - rocBLAS not used |

Runs MIOpenDriver convfp16 when present; otherwise falls back to torch.nn.functional.conv2d fp16. batch_size, input channels/height/width, output_channels, kernel_height, num_iterations, and conv_direction are used. warmup_iters applies only to the PyTorch fallback.

## 6. Installation

```bash
Run MIOpenDriver convfp16 when on PATH or /opt/rocm/bin; otherwise run a PyTorch conv2d fallback
```

## 7. Running the Benchmark

```bash
Run MIOpenDriver convfp16 when on PATH or /opt/rocm/bin; otherwise run a PyTorch conv2d fallback
```

**Validating results separately:**

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional
python3 -m venv .venv
source ".venv/bin/activate"
".venv/bin/python" scripts/validate_results.py
```

## 8. Output

### `results/benchmark.db` (SQLite)

MIOpenDriver stdout, or a PyTorch fallback CSV. The header includes kernel_time_ms and direction

name,n,c,ho,wo,y,x,k,flopCnt,bytesRead,bytesWritten,GFLOPs,GB/s,timeMs,kernel_time_ms,direction
fwd-conv,1,64,56,56,3,3,64,1,1,1,12.0,1.0,1.5,1.5,fwd

```bash
Run MIOpenDriver convfp16 when on PATH or /opt/rocm/bin; otherwise run a PyTorch conv2d fallback
```

### `results/summary.json`

Consolidated metrics from the most recent run — suitable for CI artifact upload or dashboard ingestion.

### `results/raw/<timestamp>.txt`

MIOpenDriver stdout, or a PyTorch fallback CSV. The header includes kernel_time_ms and direction

name,n,c,ho,wo,y,x,k,flopCnt,bytesRead,bytesWritten,GFLOPs,GB/s,timeMs,kernel_time_ms,direction
fwd-conv,1,64,56,56,3,3,64,1,1,1,12.0,1.0,1.5,1.5,fwd

## 9. Baselines / Thresholds

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

## 10. Troubleshooting

**`setup.sh` missing collector**
Create cannot finish without `scripts/collect_workload.py`.

**`self_check` overlay rewritten**
Do not overwrite files listed in `results/overlay_lock.json`.

**Remote SSH drop during setup**
Reconnect and resume `bash setup.sh --assume-yes`. Do not wipe `.venv` or `.cache`.

## 11. NVIDIA H100 Coding Differences

Primary target is AMD ROCm. NVIDIA notes in this section are reference only and are not the execution path.

## Repository layout

```text
.
├── setup.sh
├── run_benchmark.sh
├── benchmark_specification.json
├── .github/workflows/      # thin CI callers (see Continuous Integration)
├── config/
├── scripts/
├── src/
├── tests/
├── docs/
├── results/
└── LICENSE
```

## Continuous Integration

| Workflow | Runs on | When | What it does |
|---|---|---|---|
| [CI](.github/workflows/ci.yml) | GitHub-hosted runner | every pull request, and every push to `main` | shellcheck, ruff, `bash -n`, `compileall`, `run_benchmark.sh --help`, specification schema, the results validator on a seeded fixture, required files, and actionlint. No GPU and no benchmark run. |
| [GPU Smoke Benchmark](.github/workflows/gpu-smoke.yml) | self-hosted runner labeled `gpu`, `amd`, `ubu2604` | only when started by hand: **Actions → GPU Smoke Benchmark → Run workflow** (choose `smoke`, `baseline` or `extended`) | Verifies the pre-provisioned GPU stack, records `results/environment.json` (driver, runtime, kernel, GPU), runs the profile with `--validate`, shows headline metrics on the run page, and uploads the results. |

Both files are short callers. The steps themselves live once, for every workload in the suite, in [`garys-gpu-benchmarks/shared-workflows`](https://github.com/garys-gpu-benchmarks/shared-workflows), pinned at `@v1`. The GPU workflow is never triggered by pull requests, so code from a fork cannot run on the GPU host.

### Running it as part of the AMD Ubuntu 26.04 bundle

This repository is one of the 32 workloads in [`bundle-amd-ubuntu-2604`](https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2604), which holds them as git submodules. To put the whole bundle on a GPU host and run this workload from it:

```bash
git clone --recurse-submodules https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2604 /opt/benchmarks
cd /opt/benchmarks/318-gpu-bench-amd-miopen-convolution-micro-ubu2604
bash run_benchmark.sh --profile smoke --validate
```
