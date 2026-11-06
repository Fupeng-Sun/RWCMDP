"""Helpers shared by the experiment scripts: import path, argument parsing, parallel execution with checkpoints."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one thread per worker process

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DEFAULT_OUT = os.path.join(ROOT, "output")


def parser(description, outputs=True):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    if outputs:
        p.add_argument("--workers", type=int, default=6, help="number of worker processes (default 6)")
        p.add_argument("--out-dir", default=DEFAULT_OUT, help="directory for the result files (default output/)")
    return p


def run_tasks(fn, tasks, workers, out_path, label=str):
    """Run fn(*task) for every task in parallel; the records are written to out_path after every finished task."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    results, t0 = [], time.time()
    print(f"{len(tasks)} tasks, {workers} workers -> {out_path}", flush=True)
    if workers <= 1:
        for n, t in enumerate(tasks, 1):
            results.append(fn(*t))
            _dump(results, out_path)
            print(f"{n}/{len(tasks)} done {label(t)}, {time.time() - t0:.0f}s", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futures = {ex.submit(fn, *t): t for t in tasks}
        for n, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            _dump(results, out_path)
            print(f"{n}/{len(tasks)} done {label(futures[f])}, {time.time() - t0:.0f}s", flush=True)
    return results


def _dump(results, path):
    with open(path, "wb") as f:
        pickle.dump(results, f)
