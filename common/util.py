"""Small helpers used by both the ViT and DiT harnesses.

Deliberately torch-free and numpy-free, so importing them costs nothing and
pulls in no framework state (thread counts, CUDA context) as a side effect.
"""


def percentile(values, p):
    """Linear-interpolated percentile; no numpy dependency.

    Matches vllm_dit_vit_benchmark.py's implementation so percentiles are
    computed identically across every harness in this repo.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + frac * (ordered[high] - ordered[low])


def slope(points):
    """Least-squares slope of y over x for [(x, y), ...]. Zero if degenerate.

    The serving harnesses fit it to queue depth over time: a backlog that
    trends upward never reached steady state.
    """
    if len(points) < 2:
        return 0.0
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    denom = sum((x - mx) ** 2 for x, _ in points)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in points) / denom


def sync(device):
    """Block until queued work on `device` has finished; no-op on CPU.

    torch is imported lazily so the rest of this module stays import-free.
    """
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()
