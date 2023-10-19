import torch
import gc


def b2mb(x):
    """Convert bytes to megabytes."""
    return int(x / 2**20)


class TorchTracemalloc:
    """
    This context manager is used to track the peak memory usage of the process
    """
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # reset the peak gauge to zero [imp to compare relative memory usage]
        self.begin = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *exc):
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

