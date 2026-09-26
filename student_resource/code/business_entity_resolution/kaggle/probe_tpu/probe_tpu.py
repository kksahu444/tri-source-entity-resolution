"""Kaggle machine probe: CPU cores, RAM, disk, preinstalled packages. Downloads no data."""
import importlib, os, shutil, subprocess
print("nproc:", os.cpu_count())
print(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout)
print("disk /kaggle/tmp GB free:", round(shutil.disk_usage("/kaggle").free / 2**30, 1) if os.path.exists("/kaggle") else "n/a")
for m in ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "lightgbm", "xgboost", "rapidfuzz", "numba", "sparse_dot_topn", "anyascii", "yaml", "torch"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
