"""ViT image-classification benchmarks.

  vit_benchmark         accuracy pass, or offline throughput sweep (--throughput)
  server_vit_benchmark  request-rate ramp against a latency SLA
  diag_vit_inference    where server_vit_benchmark's inference time goes
  catalog               model list and dataset shared by the three

Run from the repository root, e.g. python -m vit.vit_benchmark --help
"""
