import os
import logging

logger = logging.getLogger("openclaw.device")


class DeviceManager:
    """Auto-detects and manages compute devices (GPU/MPS/CPU).
    Pattern from PyTorch (100k stars): unified device management."""

    def __init__(self):
        self._device = None
        self._device_name = None
        self._vram_gb = 0.0
        self._backend = "cpu"
        self._detect()

    def _detect(self):
        """Auto-detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                self._device_name = torch.cuda.get_device_name(0)
                self._vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
                self._backend = "cuda"
                logger.info(f"CUDA device: {self._device_name} ({self._vram_gb:.1f} GB)")
                return
            if hasattr(torch, "mps") and torch.mps.is_available():
                self._device = torch.device("mps")
                self._backend = "mps"
                self._device_name = "Apple MPS"
                logger.info("MPS device available")
                return
        except ImportError:
            pass
        self._device_name = "CPU"
        self._backend = "cpu"
        logger.info("No GPU detected, using CPU")

    @property
    def device(self):
        if self._device is None:
            self._detect()
        return self._device

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def vram_gb(self) -> float:
        return self._vram_gb

    @property
    def name(self) -> str:
        if self._device_name is None:
            self._detect()
        return self._device_name

    @property
    def is_cuda(self) -> bool:
        return self._backend == "cuda"

    @property
    def is_mps(self) -> bool:
        return self._backend == "mps"

    @property
    def is_cpu(self) -> bool:
        return self._backend == "cpu"

    def recommended_batch_size(self) -> int:
        if self._vram_gb >= 24:
            return 8
        if self._vram_gb >= 16:
            return 4
        if self._vram_gb >= 8:
            return 2
        return 1

    def recommended_quantization(self) -> str:
        if self._vram_gb >= 24:
            return "8bit"
        if self._vram_gb >= 12:
            return "4bit"
        return "4bit"

    def summary(self) -> str:
        return (
            f"Device: {self.name}\n"
            f"Backend: {self.backend}\n"
            f"VRAM: {self.vram_gb:.1f} GB\n"
            f"Batch: {self.recommended_batch_size()}\n"
            f"Quant: {self.recommended_quantization()}"
        )
