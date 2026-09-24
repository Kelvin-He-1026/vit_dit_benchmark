"""Model-agnostic machinery shared by the vit/ and dit/ benchmarks.

  paths      repository and per-machine output paths; sets HF_HUB_CACHE
  hostinfo   server / CPU / GPU identification for the run header
  quantize   torchao post-training quantisation recipes
  resources  CPU / GPU utilisation and power sampling
  stats      percentile and least-squares slope
  sweep      offline throughput sweep: replica pool, core splitting, flags

Nothing in here imports from vit/ or dit/, and neither of those imports from
the other.
"""
