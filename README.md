# miopen-conv-fix

PyTorch C++ extension that fixes the MIOpen workspace=0 bug for 1D convolutions on AMD ROCm GPUs.

## The Problem

PyTorch's ROCm backend passes `workspace=0` to MIOpen for convolution operations, forcing it to use `ConvDirectNaiveConvFwd` — the slowest possible solver. This causes **10-12x slowdowns** for workloads using 1D convolutions (TTS decoders, audio codecs, etc.).

Tracked upstream:
- [pytorch/pytorch#150168](https://github.com/pytorch/pytorch/issues/150168)
- [ROCm/MIOpen#3650](https://github.com/ROCm/MIOpen/issues/3650)
- [ROCm/TheRock#3077](https://github.com/ROCm/TheRock/issues/3077)

## The Fix

This extension bypasses PyTorch's conv wrapper and calls MIOpen's C API directly with proper workspace allocation. It:

1. Queries `miopenConvolutionForwardGetWorkSpaceSize()` for the required workspace
2. Allocates workspace via PyTorch's caching allocator
3. Runs `miopenFindConvolutionForwardAlgorithm()` to select the fastest algorithm
4. Executes the convolution with the optimized algorithm and workspace
5. Caches the algorithm selection for repeated calls with the same shapes

## Installation

```bash
pip install miopen-conv-fix
```

Requires ROCm and PyTorch with HIP support. On non-ROCm systems, installs as a no-op.

## Usage

```python
import miopen_conv_fix

# Monkey-patch nn.Conv1d and nn.ConvTranspose1d globally
miopen_conv_fix.patch()

# Or use as drop-in replacements
output = miopen_conv_fix.conv1d(input, weight, bias, stride=1, padding=0)
output = miopen_conv_fix.conv_transpose1d(input, weight, bias, stride=2, padding=1)
```

## Supported Operations

- `nn.Conv1d` / `F.conv1d`
- `nn.ConvTranspose1d` / `F.conv_transpose1d`
- Data types: float32, float16, bfloat16

## How It Works

On ROCm, MIOpen requires temporary workspace memory to use optimized algorithms (implicit GEMM, Winograd, etc.). PyTorch's HIP backend has a bug where it passes `workspace_size=0`, forcing MIOpen to fall back to a naive direct convolution.

This extension creates MIOpen descriptors, queries the workspace size, allocates it, and calls the convolution with proper workspace — exactly what PyTorch should be doing but isn't.

The extension is a no-op on CUDA and CPU — it only activates on ROCm.

## License

MIT
