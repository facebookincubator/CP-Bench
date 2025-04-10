# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
from enum import Enum


class AcceleratorVendor(Enum):
    NVIDIA = "NVIDIA"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def set(cls, vendor_str=None):
        """Sets the accelerator based on user input or auto-detection."""
        if vendor_str:
            try:
                cls._current = cls(vendor_str.upper())
            except ValueError:
                raise ValueError(f"Unknown accelerator vendor: {vendor_str}")
        else:
            cls._current = cls._auto_detect()

    @classmethod
    def get(cls):
        """Retrieves the selected accelerator, ensuring it's set first."""
        if not hasattr(cls, "_current") or cls._current is None:
            cls.set()  # Auto-detect if not explicitly set
        return cls._current

    @classmethod
    def _auto_detect(cls) -> "AcceleratorVendor":
        """Attempts to detect the accelerator automatically."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                if "nvidia" in gpu_name:
                    return cls.NVIDIA
        except ImportError:
            pass
        return cls.UNKNOWN
