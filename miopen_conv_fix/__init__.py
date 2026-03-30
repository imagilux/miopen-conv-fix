"""MIOpen Conv Fix — proper workspace allocation for 1D convolutions on ROCm.

Fixes pytorch/pytorch#150168 where PyTorch passes workspace=0 to MIOpen,
forcing fallback to ConvDirectNaiveConvFwd (~10-12x slower).

Uses MIOpen's Immediate Mode API with per-solution validation to avoid
segfaults from broken kernels on immature architectures (e.g. gfx1201).

Usage:
    import miopen_conv_fix
    miopen_conv_fix.patch_module(model)  # Patch conv layers in a specific module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__version__ = "0.3.0"

_IS_ROCM = (
    torch.cuda.is_available()
    and hasattr(torch.version, "hip")
    and torch.version.hip is not None
)

if _IS_ROCM:
    try:
        from miopen_conv_fix._C import (
            conv1d_forward,
            conv_transpose1d_forward,
            add_solver_blacklist,
            clear_solver_blacklist,
            clear_algo_cache,
        )

        _HAS_EXT = True
    except ImportError:
        _HAS_EXT = False
else:
    _HAS_EXT = False

# Known-bad solver IDs per GPU architecture.
# Populated as specific failures are identified. Use "*" for all architectures.
_KNOWN_BAD_SOLVERS: dict[str, list[int]] = {
    # "gfx1201": [<solution_id>, ...],
}


def apply_default_blacklist():
    """Apply known-bad solver blacklists for the current GPU."""
    if not _HAS_EXT:
        return
    for arch, sol_ids in _KNOWN_BAD_SOLVERS.items():
        for sid in sol_ids:
            add_solver_blacklist(arch, sid)


# Apply at import time so blacklists are active before any conv runs
if _HAS_EXT:
    apply_default_blacklist()


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


def _make_conv1d_forward(orig_forward):
    """Create a patched forward method for nn.Conv1d with fallback."""
    def forward(self, input):
        try:
            s = [self.stride[0]]
            p = [self.padding[0]]
            d = [self.dilation[0]]
            return conv1d_forward(input, self.weight, self.bias, s, p, d, self.groups)
        except RuntimeError as e:
            import logging
            logging.getLogger("miopen_conv_fix").debug(
                f"Fallback to PyTorch conv: {e}"
            )
            return orig_forward(self, input)
    return forward


def _make_convt1d_forward(orig_forward):
    """Create a patched forward method for nn.ConvTranspose1d with fallback."""
    def forward(self, input, output_size=None):
        try:
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                self.dilation,
            ) if output_size is not None else self.output_padding
            s = [self.stride[0]]
            p = [self.padding[0]]
            op_val = output_padding[0] if isinstance(output_padding, (list, tuple)) else output_padding
            op = [op_val]
            d = [self.dilation[0]]
            return conv_transpose1d_forward(input, self.weight, self.bias, s, p, op, self.groups, d)
        except RuntimeError as e:
            import logging
            logging.getLogger("miopen_conv_fix").debug(
                f"Fallback to PyTorch conv_transpose: {e}"
            )
            return orig_forward(self, input, output_size)
    return forward


def patch_module(module: nn.Module):
    """Patch Conv1d and ConvTranspose1d layers within a specific module.

    Only patches layers inside the given module, not globally. This avoids
    breaking other models (e.g., VQ quantizer) that use small Conv1d layers
    where the workspace fix isn't needed.

    Only activates on ROCm with the C extension available. No-op otherwise.
    """
    if not _HAS_EXT:
        import logging
        logging.getLogger(__name__).info(
            "miopen-conv-fix: not on ROCm or C extension not available, patch_module is a no-op"
        )
        return 0

    import types
    count = 0

    for name, child in module.named_modules():
        if isinstance(child, nn.ConvTranspose1d):
            orig = type(child).forward
            child.forward = types.MethodType(_make_convt1d_forward(orig), child)
            count += 1
        elif isinstance(child, nn.Conv1d):
            orig = type(child).forward
            child.forward = types.MethodType(_make_conv1d_forward(orig), child)
            count += 1

    import logging
    logging.getLogger(__name__).info(
        f"miopen-conv-fix: patched {count} Conv1d/ConvTranspose1d layers in {type(module).__name__}"
    )
    return count


# Keep patch() for backwards compatibility but make it target-specific
def patch():
    """DEPRECATED: Use patch_module(model) instead to avoid patching unrelated Conv1d layers."""
    import logging
    logging.getLogger(__name__).warning(
        "miopen_conv_fix.patch() patches ALL Conv1d globally which can break other models. "
        "Use miopen_conv_fix.patch_module(model) to patch a specific model instead."
    )
