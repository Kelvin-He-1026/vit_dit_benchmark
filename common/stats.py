"""Small statistics helpers shared by the server benchmarks. No numpy."""


def percentile(values, p):
    """Linear-interpolated percentile; no numpy dependency.

    Matches vllm_dit_vit_benchmark.py's implementation so percentiles are
    computed identically across the two harnesses.
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
    """Least-squares slope of y over x. Zero for degenerate input."""
    if len(points) < 2:
        return 0.0
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    denom = sum((x - mx) ** 2 for x, _ in points)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in points) / denom
