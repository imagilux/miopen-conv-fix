# miopen-conv-fix

PyTorch C++ extension that fixes the MIOpen workspace=0 bug for 1D convolutions on AMD ROCm GPUs.

## The Problem

PyTorch's ROCm backend passes `workspace=0` to MIOpen for convolution operations, forcing it to use `ConvDirectNaive` — the slowest possible solver. This causes **10-34x slowdowns** for workloads using 1D convolutions (TTS decoders, audio codecs, etc.).

Tracked upstream:
- [pytorch/pytorch#150168](https://github.com/pytorch/pytorch/issues/150168)
- [ROCm/MIOpen#3650](https://github.com/ROCm/MIOpen/issues/3650)
- [ROCm/TheRock#3077](https://github.com/ROCm/TheRock/issues/3077)

## How It Works

MIOpen ships **pre-compiled convolution kernels** (GEMM, implicit GEMM, Winograd, etc.) for each GPU architecture. These are optimized by AMD engineers and bundled in the MIOpen package. The kernels need temporary **workspace memory** to operate — without it, MIOpen falls back to `ConvDirectNaive`, a simple but slow kernel that needs zero workspace.

PyTorch's bug is that it passes `workspace_size=0`, so MIOpen never gets to use its fast kernels.

This extension bypasses PyTorch's conv wrapper and calls MIOpen's **Immediate Mode API** directly:

1. **Enumerate** — asks MIOpen for all available pre-compiled solutions for a given conv shape via `miopenConvolutionForwardGetSolution()`
2. **Select** — picks the best non-blacklisted solution by MIOpen's heuristic ranking
3. **Compile** — prepares the selected solution for execution via `miopenConvolutionForwardCompileSolution()` (loads and configures the pre-built kernel, not true compilation)
4. **Cache** — stores the solution ID + workspace size in an in-process `algo_cache` so steps 1-3 are skipped for repeated calls with the same conv shape
5. **Execute** — allocates workspace via PyTorch's caching allocator and dispatches via `miopenConvolutionForwardImmediate()`

The first decode after startup takes ~3-5s as MIOpen evaluates solutions for each unique conv shape. Subsequent calls hit the cache and run in milliseconds.

### Why Immediate Mode instead of Find API

MIOpen offers two selection strategies:

- **Find API** (`miopenFindConvolutionForwardAlgorithm`) — benchmarks candidate kernels by running them. On immature architectures (e.g., gfx1201/RDNA 4), some kernels segfault during benchmarking, crashing the process.
- **Immediate Mode API** (`miopenConvolutionForwardGetSolution`) — returns solutions ranked by heuristic without running them. Combined with a per-architecture solver blacklist, this safely avoids broken kernels.

This extension uses the Immediate Mode API for stability on new GPU architectures.

### Relationship to torch.compile / Triton / Inductor

This extension is completely independent of PyTorch's compilation stack:

| Layer | What it does | Used by miopen-conv-fix? |
|---|---|---|
| `torch.compile()` | Captures Python model into FX graph | No |
| **Inductor** | PyTorch's compilation backend, lowers FX to Triton/C++ | No |
| **Triton** | Generates GPU kernels from Python-like DSL | No |
| **MIOpen** | AMD's conv/GEMM library with pre-built kernels | **Yes** |
| **HIP** | AMD's GPU runtime (like CUDA) | **Yes** |

`torch.compile` goes: Python model → Inductor → Triton → HIP → GPU code (generated at compile time).
`miopen-conv-fix` goes: MIOpen C API → pre-built AMD kernels → HIP → GPU code (shipped with MIOpen).

On RDNA 4 (gfx1201), `torch.compile` produces **4.6x slower** kernels than eager mode due to immature Triton codegen. miopen-conv-fix achieves **34x speedup** over the naive fallback by using AMD's hand-optimized kernels directly.

## Installation

```bash
pip install miopen-conv-fix
```

Or build from source (required for Docker with ROCm):

```bash
# Install ROCm dev headers
apt-get install -y rocthrust-dev rocprim-dev miopen-hip-dev

# Build
git clone https://github.com/Imagilux/miopen-conv-fix.git
cd miopen-conv-fix
python setup.py build_ext --inplace
```

Requires ROCm and PyTorch with HIP support. On non-ROCm systems, installs as a no-op.

## Usage

```python
import miopen_conv_fix

# Patch all Conv1d/ConvTranspose1d layers in a specific model
count = miopen_conv_fix.patch_module(model)
print(f"Patched {count} conv layers")

# Or use as drop-in replacements
output = miopen_conv_fix.conv1d(input, weight, bias, stride=1, padding=0)
output = miopen_conv_fix.conv_transpose1d(input, weight, bias, stride=2, padding=1)

# Manage solver blacklist (skip known-broken solutions per GPU arch)
miopen_conv_fix.add_solver_blacklist("gfx1201", solution_id)
miopen_conv_fix.clear_solver_blacklist()

# Clear algo cache (forces re-evaluation of solutions)
miopen_conv_fix.clear_algo_cache()
```

## Supported Operations

- `nn.Conv1d` / `F.conv1d`
- `nn.ConvTranspose1d` / `F.conv_transpose1d`
- Data types: float32, float16, bfloat16

## Performance

Benchmarked on Fish Speech DAC decoder (88 conv layers) on AMD RX 9070 XT (gfx1201):

| Configuration | VQ Decode Time | Speedup |
|---|---|---|
| PyTorch native (workspace=0, ConvDirectNaive) | 7.98s | 1x |
| miopen-conv-fix (Immediate Mode, GEMM kernels) | 0.23s | **34x** |
| miopen-conv-fix streaming chunks (50 tokens) | 0.056s | **143x** |

## License

MIT
