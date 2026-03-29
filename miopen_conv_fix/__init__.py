"""MIOpen Conv Fix — proper workspace allocation for 1D convolutions on ROCm.

Fixes pytorch/pytorch#150168 where PyTorch passes workspace=0 to MIOpen,
forcing fallback to ConvDirectNaiveConvFwd (~10-12x slower).

Usage:
    import miopen_conv_fix
    miopen_conv_fix.patch()  # Monkey-patches nn.Conv1d and nn.ConvTranspose1d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__version__ = "0.1.1"

_IS_ROCM = (
    torch.cuda.is_available()
    and hasattr(torch.version, "hip")
    and torch.version.hip is not None
)

_patched = False

if _IS_ROCM:
    try:
        from miopen_conv_fix._C import conv1d_forward, conv_transpose1d_forward

        _HAS_EXT = True
    except ImportError:
        _HAS_EXT = False
else:
    _HAS_EXT = False


def conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Drop-in replacement for F.conv1d with proper MIOpen workspace on ROCm."""
    if _HAS_EXT and input.is_cuda:
        s = [stride] if isinstance(stride, int) else list(stride)
        p = [padding] if isinstance(padding, int) else list(padding)
        d = [dilation] if isinstance(dilation, int) else list(dilation)
        return conv1d_forward(input, weight, bias, s, p, d, groups)
    return F.conv1d(input, weight, bias, stride, padding, dilation, groups)


def conv_transpose1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    dilation: int = 1,
) -> torch.Tensor:
    """Drop-in replacement for F.conv_transpose1d with proper MIOpen workspace on ROCm."""
    if _HAS_EXT and input.is_cuda:
        s = [stride] if isinstance(stride, int) else list(stride)
        p = [padding] if isinstance(padding, int) else list(padding)
        op = [output_padding] if isinstance(output_padding, int) else list(output_padding)
        d = [dilation] if isinstance(dilation, int) else list(dilation)
        return conv_transpose1d_forward(input, weight, bias, s, p, op, groups, d)
    return F.conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)


# Store original forward methods for unpatching
_orig_conv1d_forward = None
_orig_convt1d_forward = None


def patch():
    """Monkey-patch nn.Conv1d and nn.ConvTranspose1d to use MIOpen with proper workspace.

    Only activates on ROCm with the C extension available. No-op on CUDA/CPU.
    """
    global _patched, _orig_conv1d_forward, _orig_convt1d_forward

    if _patched:
        return

    if not _HAS_EXT:
        import logging
        logging.getLogger(__name__).info(
            "miopen-conv-fix: not on ROCm or C extension not available, patch is a no-op"
        )
        return

    _orig_conv1d_forward = nn.Conv1d.forward
    _orig_convt1d_forward = nn.ConvTranspose1d.forward

    def _conv1d_forward(self, input):
        # .contiguous() materializes parametrized weights (e.g. weight_norm)
        # into a tensor with storage that MIOpen can access.
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None
        return conv1d(
            input, weight, bias,
            self.stride[0], self.padding[0], self.dilation[0], self.groups,
        )

    def _convt1d_forward(self, input, output_size=None):
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,
            self.dilation, # type: ignore[arg-type]
        ) if output_size is not None else self.output_padding
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None
        return conv_transpose1d(
            input, weight, bias,
            self.stride[0], self.padding[0],
            output_padding[0] if isinstance(output_padding, (list, tuple)) else output_padding,
            self.groups, self.dilation[0],
        )

    nn.Conv1d.forward = _conv1d_forward
    nn.ConvTranspose1d.forward = _convt1d_forward
    _patched = True

    import logging
    logging.getLogger(__name__).info(
        "miopen-conv-fix: patched nn.Conv1d and nn.ConvTranspose1d for MIOpen workspace fix"
    )


def unpatch():
    """Restore original nn.Conv1d and nn.ConvTranspose1d forward methods."""
    global _patched, _orig_conv1d_forward, _orig_convt1d_forward

    if not _patched:
        return

    if _orig_conv1d_forward is not None:
        nn.Conv1d.forward = _orig_conv1d_forward
    if _orig_convt1d_forward is not None:
        nn.ConvTranspose1d.forward = _orig_convt1d_forward

    _patched = False
