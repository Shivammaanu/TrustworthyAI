"""Package init.

macOS + conda: scikit-learn and PyTorch can each load their own OpenMP runtime; importing
scikit-learn first can then crash PyTorch with a segmentation fault and no Python error.
Importing torch first, allowing the duplicate runtime and fixing the thread count avoids it."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import faulthandler  # noqa: E402

faulthandler.enable()          # if the process still crashes, print where it happened

try:
    import torch  # noqa: E402,F401  (must be loaded before scikit-learn)
except ImportError:
    pass
