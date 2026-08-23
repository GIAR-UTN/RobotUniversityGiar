"""Robust CUDA-usability probe shared by the rugiar drivers.

torch.cuda.is_available() only checks that the driver can ENUMERATE a CUDA
device -- it returns True even when the driver cannot actually create a compute
context (e.g. a wedged GPU or broken GSP firmware, where cuDevicePrimaryCtxRetain
fails with CUDA_ERROR_OPERATING_SYSTEM while the device still shows up in
nvidia-smi and the display keeps working). Genesis (gs.init(backend=gs.cuda)),
Warp and torch all hit exactly this trap: the device enumerates, then context
creation fails at the driver level and every downstream allocation dies.

Every driver that decides "use CUDA or not" from an env var or a --device flag
should probe with cuda_is_usable() BEFORE handing the device to the simulator,
so an enumerated-but-unusable GPU becomes a clean CPU fallback with an
actionable message instead of a mid-init crash or a scary driver-error spew.
"""
import torch


def cuda_is_usable(device: str = "cuda:0") -> tuple[bool, str]:
    """True only if a real CUDA context can be created AND used on `device`.

    Unlike torch.cuda.is_available(), this actually allocates a tensor and
    runs a trivial kernel -- the same operations that fail with
    cudaErrorDevicesUnavailable / CUDA_ERROR_OPERATING_SYSTEM when the driver
    can enumerate the GPU but not create a compute context.

    Returns (usable, reason). On failure `reason` is a short, actionable
    message safe to print in a log line.
    """
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False (no CUDA device, or driver broken at init)"
    try:
        torch.rand(1, device=device)
        (torch.rand(1, device=device) * 2).sum().item()
        torch.cuda.synchronize(device)
    except Exception as e:
        return False, f"CUDA device {device} enumerates but cannot create a usable context: {e}"
    return True, ""
