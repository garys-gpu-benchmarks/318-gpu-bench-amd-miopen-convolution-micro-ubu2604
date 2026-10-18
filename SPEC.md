# SPEC.md: "The Human How"; Exact technical requirements, environment setup, implementation details, etc.

## Execution Description

Runs MIOpenDriver convfp16 when present; otherwise falls back to torch.nn.functional.conv2d fp16. batch_size, input channels/height/width, output_channels, kernel_height, num_iterations, and conv_direction are used. warmup_iters applies only to the PyTorch fallback. conv_direction=all runs -F 1,2,4; otherwise -F 1. Sweep dimensions: dtype, input_source, conv_direction, batch_size, input_channels, input_height, input_width, output_channels.

## Parameters

| Parameter | CLI Flag | Tested Values | Default | Description |
| --- | --- | --- | --- | --- |
| dtype | `--dtype` | smoke=FP16, baseline=FP16, extended=FP16 | FP16 | From Parameter list; see Execution Description With Parameters. |
| input_source | `--input-source` | smoke=synthetic, baseline=synthetic, extended=synthetic | synthetic | From Parameter list; see Execution Description With Parameters. |
| conv_direction | `--conv-direction` | smoke=all, baseline=all, extended=all | all | From Parameter list; see Execution Description With Parameters. |
| batch_size | `--batch-size` | smoke=1, baseline=32, extended=32 | 32 | From Parameter list; see Execution Description With Parameters. |
| input_channels | `--input-channels` | smoke=8, baseline=3, extended=3 | 3 | From Parameter list; see Execution Description With Parameters. |
| input_height | `--input-height` | smoke=16, baseline=224, extended=224 | 224 | From Parameter list; see Execution Description With Parameters. |
| input_width | `--input-width` | smoke=16, baseline=224, extended=224 | 224 | From Parameter list; see Execution Description With Parameters. |
| output_channels | `--output-channels` | smoke=8, baseline=64, extended=64 | 64 | From Parameter list; see Execution Description With Parameters. |
| kernel_height | `--kernel-height` | smoke=3, baseline=3, extended=3 | 3 | From Parameter list; see Execution Description With Parameters. |
| kernel_width | `--kernel-width` | smoke=3, baseline=3, extended=3 | 3 | From Parameter list; see Execution Description With Parameters. |
| stride_h | `--stride-h` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |
| stride_w | `--stride-w` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |
| group_count | `--group-count` | smoke=1, baseline=1, extended=1 | 1 | From Parameter list; see Execution Description With Parameters. |
| search_strategy | `--search-strategy` | smoke=find, baseline=find, extended=find | find | From Parameter list; see Execution Description With Parameters. |
| warmup_iters | `--warmup-iters` | smoke=1, baseline=5, extended=10 | 5 | From Parameter list; see Execution Description With Parameters. |
| num_iterations | `--num-iterations` | smoke=2, baseline=168000, extended=514000 | 168000 | From Parameter list; see Execution Description With Parameters. |

## Invocation

```bash
Run MIOpenDriver convfp16 when on PATH or /opt/rocm/bin; otherwise run a PyTorch conv2d fallback
```

## Raw Output Format

MIOpenDriver stdout, or a PyTorch fallback CSV. The header includes kernel_time_ms and direction

name,n,c,ho,wo,y,x,k,flopCnt,bytesRead,bytesWritten,GFLOPs,GB/s,timeMs,kernel_time_ms,direction
fwd-conv,1,64,56,56,3,3,64,1,1,1,12.0,1.0,1.5,1.5,fwd

## Metrics

- **#1: Convolution kernel execution time, forward** — stored as `kernel_time_ms`.
- **#2: Modeled compute throughput, forward, GFLOP/s** — stored as `modeled_gflops_per_s`.
- **#3: Minimum operand traffic, GB/s, forward** — stored as `min_operand_traffic_gb_s`.
- **#4: FLOP count** — stored as `flopCnt`.

## Framework

Runs MIOpenDriver convfp16 when present; otherwise falls back to torch.nn.functional.conv2d fp16. batch_size, input channels/height/width, output_channels, kernel_height, num_iterations, and conv_direction are used. warmup_iters applies only to the PyTorch fallback.

## Installation and Execution Summary

Run MIOpenDriver convfp16 with yaml batch, channels, spatial size, kernel, and iteration counts, or fall back to PyTorch fp16 conv2d if the driver is missing, to measure convolution time and GFLOPs. Do not map search_strategy=find to MIOpenDriver -S 1

## Platform Portability

- **AMD (primary):** ```bash
Run MIOpenDriver convfp16 when on PATH or /opt/rocm/bin; otherwise run a PyTorch conv2d fallback
```
- **NVIDIA:** Primary target is AMD ROCm. NVIDIA notes in this section are reference only and are not the execution path.

## Model Context Protocols

- **Active:** None

## Execution-Loop Validation Contract

EXECUTION CHAIN: `run_benchmark.sh` ➔ raw output ➔ `scripts/parse_results.py` ➔ `results/benchmark.db` ➔ `scripts/validate_results.py`

This benchmark uses a lightweight, SQLite-integrated execution loop for result validation. All validation is performed by `scripts/validate_results.py`.

### Validation script usage

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional; select the installed interpreter

# After a live run:
".venv/bin/python" scripts/validate_results.py --db results/benchmark.db

# CI / no-GPU path (seeds fixture and validates it):
".venv/bin/python" scripts/validate_results.py --seed-fixture --quiet

# Override DB path via environment variable:
BENCHMARK_DB=tests/fixtures/benchmark.db \
  ".venv/bin/python" scripts/validate_results.py
```

### Run artifact contract

MIOpenDriver stdout, or a PyTorch fallback CSV. The header includes kernel_time_ms and direction

name,n,c,ho,wo,y,x,k,flopCnt,bytesRead,bytesWritten,GFLOPs,GB/s,timeMs,kernel_time_ms,direction
fwd-conv,1,64,56,56,3,3,64,1,1,1,12.0,1.0,1.5,1.5,fwd

```bash
bash run_benchmark.sh --help
bash run_benchmark.sh --profile smoke --validate
bash run_benchmark.sh --profile baseline --validate
bash run_benchmark.sh --profile extended --validate
```
`run_benchmark.sh --help` prints usage and exits. The harness calls `scripts/ensure_setup.sh` when `.setup_state` is absent.

### Required integrity checks (built into `validate_results.py`)

1. Latest run exists and `runs.status = 'ok'`.
2. `run.error_message` is NULL.
3. `started_at` and `finished_at` are valid ISO-8601 UTC strings.
4. All required aggregate metrics in `runs` are non-NULL and finite.
5. All required aggregate metrics are physically sensible (positive values). Runs MIOpenDriver convfp16 when present; otherwise falls back to torch.nn.functional.conv2d fp16. batch_size, input channels/height/width, output_channels, kernel_height, num_iterations, and conv_direction are used. warmup_iters applies only to the PyTorch fallback.
6. At least 2 sample rows exist for the latest `run_id` (sweep coverage).
7. No sample has `status = 'error'`.
8. Runs MIOpenDriver convfp16 when present; otherwise falls back to torch.nn.functional.conv2d fp16. batch_size, input channels/height/width, output_channels, kernel_height, num_iterations, and conv_direction are used. warmup_iters applies only to the PyTorch fallback.

### Baseline / Threshold configuration (`config/benchmark_config.yaml`)

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

Threshold key suffixes encode comparison direction when `thresholds:` is present: `_min` → observed value must be ≥ threshold. `_max` → observed value must be ≤ threshold. Informational `baselines:` ranges are not pass/fail gates.
