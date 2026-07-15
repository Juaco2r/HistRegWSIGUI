"""Hardware detection and registration-device configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CUDAInfo:
    """Result of checking whether CUDA can be used by the current PyTorch build."""

    available: bool
    compiled_with_cuda: bool
    torch_cuda_version: str | None
    device_names: tuple[str, ...]
    reason: str

    @property
    def device_count(self) -> int:
        return len(self.device_names)


def detect_cuda(torch_module: Any | None = None, *, probe: bool = False) -> CUDAInfo:
    """Inspect the installed PyTorch runtime and available NVIDIA devices.

    ``probe=True`` performs a tiny allocation on ``cuda:0``. This is useful for
    the explicit GUI check because it catches driver/runtime failures that a
    simple ``torch.cuda.is_available()`` call may not fully expose.
    """

    try:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]
    except Exception as exc:
        return CUDAInfo(
            available=False,
            compiled_with_cuda=False,
            torch_cuda_version=None,
            device_names=(),
            reason=f"PyTorch could not be imported: {exc}",
        )

    try:
        cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        compiled_with_cuda = bool(cuda_version)

        if not compiled_with_cuda:
            return CUDAInfo(
                available=False,
                compiled_with_cuda=False,
                torch_cuda_version=None,
                device_names=(),
                reason="This application contains a CPU-only PyTorch runtime.",
            )

        if not bool(torch_module.cuda.is_available()):
            return CUDAInfo(
                available=False,
                compiled_with_cuda=True,
                torch_cuda_version=str(cuda_version),
                device_names=(),
                reason=(
                    "The application supports CUDA, but no compatible NVIDIA GPU/driver "
                    "is currently available."
                ),
            )

        count = int(torch_module.cuda.device_count())
        names = tuple(str(torch_module.cuda.get_device_name(i)) for i in range(count))

        if count < 1:
            return CUDAInfo(
                available=False,
                compiled_with_cuda=True,
                torch_cuda_version=str(cuda_version),
                device_names=(),
                reason="CUDA was reported as available, but no CUDA device was enumerated.",
            )

        if probe:
            test_tensor = torch_module.empty(1, device="cuda:0")
            del test_tensor
            synchronize = getattr(torch_module.cuda, "synchronize", None)
            if callable(synchronize):
                synchronize()

        return CUDAInfo(
            available=True,
            compiled_with_cuda=True,
            torch_cuda_version=str(cuda_version),
            device_names=names,
            reason="CUDA is ready.",
        )
    except Exception as exc:
        return CUDAInfo(
            available=False,
            compiled_with_cuda=bool(
                getattr(getattr(torch_module, "version", None), "cuda", None)
            ),
            torch_cuda_version=getattr(getattr(torch_module, "version", None), "cuda", None),
            device_names=(),
            reason=f"CUDA check failed: {exc}",
        )


def configure_registration_device(parameters: Any, device: str) -> Any:
    """Return a deep copy of DeeperHistReg parameters configured for a device.

    DeeperHistReg stores device information in several nested dictionaries.
    This function updates every ``device`` key, every ``cuda`` boolean and any
    remaining CUDA device string so CPU mode cannot accidentally initialize a
    GPU. For CUDA mode, the selected device is propagated consistently.
    """

    normalized_device = "cuda:0" if str(device).lower().startswith("cuda") else "cpu"
    cuda_enabled = normalized_device.startswith("cuda")

    def transform(value: Any, key: Any = None) -> Any:
        key_lower = str(key).lower() if key is not None else ""

        if key_lower == "device":
            return normalized_device
        if key_lower == "cuda":
            return cuda_enabled

        if isinstance(value, dict):
            return {k: transform(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [transform(v) for v in value]
        if isinstance(value, tuple):
            return tuple(transform(v) for v in value)
        if isinstance(value, str) and value.lower().startswith("cuda"):
            return normalized_device
        return value

    return transform(deepcopy(parameters))


def format_cuda_summary(info: CUDAInfo) -> str:
    """Create a compact user-facing CUDA status string."""

    if info.available:
        devices = ", ".join(info.device_names)
        return f"CUDA {info.torch_cuda_version}: {devices}"
    if info.compiled_with_cuda:
        return f"CUDA build detected, unavailable: {info.reason}"
    return info.reason
