"""Demo: switching ptspy's numeric backend from numpy to cupy via np_wrapper.

Run with numpy only (always works):
    python examples/backend_switch_demo.py

Run with cupy (requires CUDA GPU + cupy installed):
    python examples/backend_switch_demo.py --cupy
"""
import argparse
from ptspy import np_wrapper as np
from ptspy import P, P2, steps, gridPoints

def showInfo(label, arr):
    """Print array backend, shape, and values."""
    backend = type(arr).__module__
    print(f"  {label}: module={backend}  type={type(arr).__name__}  shape={arr.shape}")
    print(f"    values={np.asnumpy(arr).tolist()}")

def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cupy", action="store_true", help="Switch backend to cupy")
    args = parser.parse_args()

    # --- numpy (default) ---
    print("=== numpy backend ===")
    np.reset("numpy")

    pts = P(10, 20, 30)
    showInfo("P(10,20,30)", pts)

    grid = P2([0, 100], [0, 100])
    showInfo("P2 grid", grid.reshape(-1, 2))

    s = steps(0, 100, 5)
    showInfo("steps(0,100,5)", s)

    gp = gridPoints([3, 3], [50, 50])
    showInfo("gridPoints 3x3", gp)

    if not args.cupy:
        print("\nPass --cupy to see the backend switch in action.")
        return

    # --- cupy ---
    print("\n=== cupy backend ===")
    np.reset("cupy")

    pts = P(10, 20, 30)
    showInfo("P(10,20,30)", pts)

    grid = P2([0, 100], [0, 100])
    showInfo("P2 grid", grid.reshape(-1, 2))

    s = steps(0, 100, 5)
    showInfo("steps(0,100,5)", s)

    gp = gridPoints([3, 3], [50, 50])
    showInfo("gridPoints 3x3", gp)

    # Demonstrate explicit transfers
    print("\n=== explicit transfers ===")
    gpu_arr = np.array([1.0, 2.0, 3.0])           # lives on GPU
    cpu_arr = np.asnumpy(gpu_arr)                  # GPU -> CPU
    back_on_gpu = np.fromcpu(cpu_arr)              # CPU -> GPU
    showInfo("gpu_arr", gpu_arr)
    showInfo("cpu_arr (numpy)", cpu_arr)
    showInfo("back_on_gpu", back_on_gpu)

    # Override example
    print("\n=== override example ===")
    _original_linspace = np.linspace
    def tracedLinspace(*args, **kwargs):
        print("    [trace] linspace called")
        return _original_linspace(*args, **kwargs)
    np.override("linspace", tracedLinspace)
    result = np.linspace(0, 1, 5)
    showInfo("linspace(0,1,5)", result)

    # Switch back
    print("\n=== back to numpy ===")
    np.reset("numpy")
    pts = P(10, 20, 30)
    showInfo("P(10,20,30)", pts)

if __name__ == "__main__":
    demo()
