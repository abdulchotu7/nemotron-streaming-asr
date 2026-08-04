"""System-level monitoring: CPU %, memory, macOS unified memory, MLX device.

CPU % is derived from ``getrusage`` CPU-time deltas over wall time; if
``psutil`` is installed it is used for current RSS, otherwise only the process
peak RSS (``ru_maxrss``, bytes on macOS) is reported. Unified memory comes from
the standard macOS ``host_statistics64`` API; MLX memory comes from
``mx.get_active_memory()`` etc. (MLX exposes no GPU utilization percentage).
"""

import ctypes
import resource
import time
from ctypes import POINTER, Structure, byref, c_int, c_uint

import mlx.core as mx


class _VMStatistics64(Structure):
    """mach/vm_statistics.h vm_statistics64 (host_statistics64, HOST_VM_INFO64)."""

    _fields_ = [
        ("free_count", c_uint),
        ("active_count", c_uint),
        ("inactive_count", c_uint),
        ("wire_count", c_uint),
        ("zero_fill_count", c_uint),
        ("reactivations", c_uint),
        ("pageins", c_uint),
        ("pageouts", c_uint),
        ("faults", c_uint),
        ("cow_faults", c_uint),
        ("lookups", c_uint),
        ("hits", c_uint),
        ("purges", c_uint),
        ("purgeable_count", c_uint),
        ("speculative_count", c_uint),
        ("decompressions", c_uint),
        ("compressions", c_uint),
        ("swapins", c_uint),
        ("swapouts", c_uint),
        ("compressor_page_count", c_uint),
        ("throttled_count", c_uint),
        ("external_page_count", c_uint),
        ("internal_page_count", c_uint),
        ("total_uncompressed_pages_in_compressor", c_uint),
    ]


def macos_unified_memory_used_bytes():
    """System-wide unified memory in use (bytes) via host_statistics64.

    Uses the Activity-Monitor-equivalent definition: active + wired +
    compressed pages (speculative is not included; it is repurposed on current
    macOS and can report bogus values). Returns None if unavailable.
    """
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        libc.mach_host_self.restype = c_uint
        host = libc.mach_host_self()
        stats = _VMStatistics64()
        count = c_uint(ctypes.sizeof(_VMStatistics64) // ctypes.sizeof(c_uint))
        libc.host_statistics64.restype = c_int
        libc.host_statistics64.argtypes = [
            c_uint, c_int, POINTER(_VMStatistics64), POINTER(c_uint),
        ]
        if libc.host_statistics64(host, 4, byref(stats), byref(count)) != 0:
            return None
        page_size = libc.sysconf(29)  # _SC_PAGESIZE
        pages = stats.active_count + stats.wire_count + stats.compressor_page_count
        return pages * page_size
    except Exception:
        return None


def mlx_memory_info():
    """MLX device memory (bytes) and device info; {} if unavailable."""
    info = {}
    try:
        info["mlx_active_bytes"] = mx.get_active_memory()
        info["mlx_peak_bytes"] = mx.get_peak_memory()
        info["mlx_cache_bytes"] = mx.get_cache_memory()
        try:
            device = mx.device_info()
            info["mlx_total_bytes"] = device.get("memory_size")
            info["gpu_name"] = device.get("device_name")
        except Exception:
            pass
    except Exception:
        pass
    return info


class SystemMonitor:
    """Samples process CPU %, RSS, peak RSS, unified memory and MLX memory."""

    def __init__(self):
        self._cpu_last = None  # (wall_s, user+sys seconds)
        self._psutil = None
        try:
            import psutil  # optional: current RSS + CPU times

            self._psutil = psutil.Process()
        except Exception:
            self._psutil = None
        self.sample()  # prime the CPU baseline so the first delta is valid

    def sample(self):
        now = time.perf_counter()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        cpu_user_sys = ru.ru_utime + ru.ru_stime

        cpu_pct = 0.0
        if self._cpu_last is not None:
            dt = now - self._cpu_last[0]
            if dt > 0:
                cpu_pct = 100.0 * (cpu_user_sys - self._cpu_last[1]) / dt
        self._cpu_last = (now, cpu_user_sys)

        rss_bytes = None
        if self._psutil is not None:
            try:
                rss_bytes = self._psutil.memory_info().rss
            except Exception:
                rss_bytes = None

        info = {
            "cpu_percent": cpu_pct,
            "rss_bytes": rss_bytes,
            "peak_rss_bytes": ru.ru_maxrss,  # bytes on macOS
            "unified_used_bytes": macos_unified_memory_used_bytes(),
        }
        info.update(mlx_memory_info())
        return info
