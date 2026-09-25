"""Summarize the two Nsight Systems decode profiles (eager launch vs CUDA graphs).

Expects the CSVs produced by:
    nsys stats --force-export true -q -r nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum \
        --filter-nvtx decode_steps --format csv results/nsys_decode_<eager|graphs>.nsys-rep > results/nsys_<eager|graphs>.csv
(record the reps with benchmarks/profile_decode.py; use --cuda-graph-trace=node for the graphs run
so kernels inside graphs are attributed individually). Writes results/nsys_summary.json.
"""
import csv
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH_APIS = ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx", "cudaGraphLaunch")


def load(name):
    with open(os.path.join(ROOT, "results", f"nsys_{name}.csv")) as f:
        nvtx, kern, api = [list(csv.DictReader(io.StringIO(t))) for t in f.read().strip().split("\n\n")[:3]]
    steps = next(int(r["Instances"]) for r in nvtx if r["Range"].endswith("forward"))
    total_ns = next(float(r["Total Time (ns)"]) for r in nvtx if r["Range"].endswith("decode_steps"))
    gpu_ns = sum(int(r["Total Time (ns)"]) for r in kern)
    launch = [r for r in api if r["Name"] in LAUNCH_APIS]
    return {
        "steps": steps,
        "wall_ms_per_step": total_ns / steps / 1e6,
        "gpu_kernel_ms_per_step": gpu_ns / steps / 1e6,
        "gpu_busy_fraction": gpu_ns / total_ns,
        "kernels_per_step": sum(int(r["Instances"]) for r in kern) / steps,
        "launch_api_calls_per_step": sum(int(r["Num Calls"]) for r in launch) / steps,
        "launch_api_ms_per_step": sum(int(r["Total Time (ns)"]) for r in launch) / steps / 1e6,
        "top_kernels": [{"name": r["Name"][:90], "pct": float(r["Time (%)"])} for r in kern[:5]],
    }


if __name__ == "__main__":
    e, g = load("eager"), load("graphs")
    out = {"batch": 32, "prompt_len": 512, "eager": e, "graphs": g,
           "latency_reduction_pct": 100 * (1 - g["wall_ms_per_step"] / e["wall_ms_per_step"])}
    for k in ("wall_ms_per_step", "gpu_kernel_ms_per_step", "gpu_busy_fraction", "kernels_per_step",
              "launch_api_calls_per_step", "launch_api_ms_per_step"):
        print(f"{k:28s} eager {e[k]:9.2f}   graphs {g[k]:9.2f}")
    print(f"decode step latency reduction under nsys: {out['latency_reduction_pct']:.1f}%")
    with open(os.path.join(ROOT, "results", "nsys_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
