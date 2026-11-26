"""Profile and optimize the P300 network from Machine_Learning_to_Analyze_EEG_Data.ipynb.

Run in Colab (Runtime -> Run all is not needed):
    !git clone https://github.com/serjankaur05/ML-for-EEG-Data
    !python ML-for-EEG-Data/profile_model.py

It measures the notebook's original training loop and inference against an
optimized version, profiles both with torch.profiler, and prints a summary.
Every number in the summary comes from this run on your hardware.

What is compared
  Training (same model, data, optimizer, seed and iteration count):
    baseline   the notebook's loop: model in eager mode, loss_data.append(loss)
               (keeps each loss tensor and its autograd node chain referenced),
               zero_grad() writes zeros into every gradient
    fused_adam same eager model, but Adam(fused=True) on GPU: one kernel updates
               every parameter instead of a chain of small per-op kernels (the
               profiler showed Adam at 46% of GPU time on a T4)
    optimized  fused Adam + torch.compile(model) (TorchInductor fuses each
               Linear's bias add and activation), loss.item() instead of the
               tensor, zero_grad(set_to_none=True)
  Inference (batch of 256 samples):
    baseline   model(x) with autograd recording, as in the notebook's test cells
    optimized  torch.inference_mode(); torch.compile; and on GPU
               torch.compile(mode="reduce-overhead"), which replays the whole
               forward pass from one CUDA graph
  If torch.compile is unavailable (no C++ compiler), the optimized runs fall
  back to eager mode and the summary says so.
"""
import copy
import gc
import json
import statistics
import time

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

torch.manual_seed(100)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same sizes as the notebook.
EEG_SAMPLE_COUNT, EEG_SAMPLE_LENGTH = 240, 226
HIDDEN1, HIDDEN2, HIDDEN3 = 500, 1000, 100
LEARNING_RATE, ITERATIONS = 1e-3, 500


def build_model() -> nn.Sequential:
    torch.manual_seed(100)
    m = nn.Sequential()
    m.add_module("Input Linear", nn.Linear(EEG_SAMPLE_LENGTH, HIDDEN1))
    m.add_module("Input Activation", nn.CELU())
    m.add_module("Hidden Linear", nn.Linear(HIDDEN1, HIDDEN2))
    m.add_module("Hidden Activation", nn.ReLU())
    m.add_module("Hidden Linear2", nn.Linear(HIDDEN2, HIDDEN3))
    m.add_module("Hidden Activation2", nn.ReLU())
    m.add_module("Hidden Linear3", nn.Linear(HIDDEN3, 10))
    m.add_module("Hidden Activation3", nn.ReLU())
    m.add_module("Output Linear", nn.Linear(10, 1))
    m.add_module("Output Activation", nn.Sigmoid())
    return m.to(DEVICE)


def sample_data():
    """The notebook's synthetic training set: positives centred at 0.5,
    negatives at 0.125 and 0.875."""
    g = torch.Generator().manual_seed(100)
    half, quarter = EEG_SAMPLE_COUNT // 2, EEG_SAMPLE_COUNT // 4
    pos = torch.rand(half, EEG_SAMPLE_LENGTH, generator=g) * 0.50 + 0.25
    low = torch.rand(quarter, EEG_SAMPLE_LENGTH, generator=g) * 0.25
    high = torch.rand(quarter, EEG_SAMPLE_LENGTH, generator=g) * 0.25 + 0.75
    x = torch.cat([pos, low, high]).to(DEVICE)
    y = torch.cat([torch.ones(half, 1), torch.zeros(half, 1)]).to(DEVICE)
    return x, y


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def train(model, x, y, iterations, optimized: bool, fused_adam: bool = False):
    loss_fn = nn.MSELoss()
    fused = fused_adam and DEVICE == "cuda"
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, fused=fused)
    loss_data = []
    for _ in range(iterations):
        out = model(x)
        loss = loss_fn(out, y)
        if optimized:
            loss_data.append(loss.item())          # plain float, graph freed
            opt.zero_grad(set_to_none=True)
        else:
            loss_data.append(loss)                 # notebook: tensor + graph kept
            opt.zero_grad()
        loss.backward()
        opt.step()
    return loss_data


def time_training(optimized: bool, repeats: int = 5, fused_adam: bool = False, compile_model: bool = None):
    """Median wall time of ITERATIONS training steps, restarting from the same
    initial weights each repeat. The optimized model is compiled and warmed up
    once before timing, so compilation time is excluded (and reported in notes)."""
    x, y = sample_data()
    base = build_model()
    init = copy.deepcopy(base.state_dict())
    model, note = base, ""
    if compile_model is None:
        compile_model = optimized
    if compile_model:
        try:
            t = time.perf_counter()
            model = torch.compile(base)
            train(model, x, y, 3, optimized=True, fused_adam=fused_adam)  # triggers compilation
            note = f"train compile time {time.perf_counter() - t:.1f}s (excluded)"
        except Exception as e:                             # e.g. no C++ compiler
            model, note = base, f"torch.compile unavailable for training: {type(e).__name__}"
    times = []
    for _ in range(repeats):
        base.load_state_dict(init)                         # compiled wrapper shares these params
        gc.collect()
        sync()
        t = time.perf_counter()
        losses = train(model, x, y, ITERATIONS, optimized, fused_adam)
        sync()
        times.append(time.perf_counter() - t)
    last = losses[-1]
    last = last.detach().item() if isinstance(last, torch.Tensor) else last
    return statistics.median(times), float(last), note


def time_inference(model, x, mode: str, repeats: int = 100):
    if mode == "baseline":
        fn = lambda: model(x)                      # noqa: E731
    else:
        def fn():
            with torch.inference_mode():
                return model(x)
    for _ in range(10):
        fn()
    sync()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1e6)
    return statistics.median(ts)


def top_ops(fn, n=8):
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE == "cuda" else [])
    with profile(activities=acts, record_shapes=True) as prof:
        fn()
    key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
    return prof, prof.key_averages().table(sort_by=key, row_limit=n)


def main():
    print(f"device={DEVICE}  torch={torch.__version__}  threads={torch.get_num_threads()}")

    # ---------------- training ----------------
    base_t, base_loss, _ = time_training(optimized=False)
    fadam_t, _, _ = time_training(optimized=False, fused_adam=True)
    opt_t, opt_loss, train_note = time_training(optimized=True, fused_adam=True)

    # ---------------- inference ----------------
    model = build_model().eval()
    x = torch.rand(256, EEG_SAMPLE_LENGTH, device=DEVICE)
    inf_base = time_inference(model, x, "baseline")
    inf_mode = time_inference(model, x, "inference_mode")
    inf_compiled, compile_note = None, ""
    try:
        compiled = torch.compile(model)
        with torch.inference_mode():
            ref, got = model(x), compiled(x)
        assert torch.allclose(ref, got, atol=1e-5), "compiled output differs"
        inf_compiled = time_inference(compiled, x, "inference_mode")
    except Exception as e:                         # e.g. no C++ compiler available
        compile_note = f"torch.compile unavailable here: {type(e).__name__}"
    inf_graphs = None
    if DEVICE == "cuda" and not compile_note:
        try:
            graphed = torch.compile(model, mode="reduce-overhead")
            with torch.inference_mode():
                for _ in range(3):                 # record the CUDA graph
                    got = graphed(x)
                assert torch.allclose(model(x), got, atol=1e-5), "CUDA graph output differs"
            inf_graphs = time_inference(graphed, x, "inference_mode")
        except Exception as e:
            compile_note += f" reduce-overhead failed: {type(e).__name__}: {e}"[:200]

    # ---------------- profiler traces ----------------
    xs, ys = sample_data()
    prof, table = top_ops(lambda: train(build_model(), xs, ys, 50, optimized=False))
    prof.export_chrome_trace("eeg_train_baseline_trace.json")
    print("\nTop ops, 50 baseline training iterations (torch.profiler):")
    print(table)
    if DEVICE == "cuda":
        _, table2 = top_ops(lambda: train(build_model(), xs, ys, 50, optimized=False, fused_adam=True))
        print("\nTop ops, 50 iterations with Adam(fused=True):")
        print(table2)

    best_inf = min(v for v in (inf_mode, inf_compiled, inf_graphs) if v is not None)
    summary = {
        "device": DEVICE,
        "torch": torch.__version__,
        "train_iterations": ITERATIONS,
        "train_baseline_s": round(base_t, 4),
        "train_optimized_s": round(opt_t, 4),
        "train_speedup": round(base_t / opt_t, 2),
        "train_fused_adam_only_s": round(fadam_t, 4),
        "train_fused_adam_speedup": round(base_t / fadam_t, 2),
        "final_loss_baseline": round(base_loss, 6),
        "final_loss_optimized": round(opt_loss, 6),
        "inference_batch": 256,
        "inference_baseline_us": round(inf_base, 1),
        "inference_mode_us": round(inf_mode, 1),
        "inference_compiled_us": None if inf_compiled is None else round(inf_compiled, 1),
        "inference_cudagraphs_us": None if inf_graphs is None else round(inf_graphs, 1),
        "inference_speedup_best": round(inf_base / best_inf, 2),
        "notes": [n for n in (train_note, compile_note) if n],
    }
    print("\n==== SUMMARY (paste this back) ====")
    print(json.dumps(summary, indent=2))
    print("trace written to eeg_train_baseline_trace.json (open in https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
