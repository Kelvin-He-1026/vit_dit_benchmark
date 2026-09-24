"""Repository paths, shared by every benchmark script.

Import this before huggingface_hub / datasets / transformers / diffusers:
importing it sets HF_HUB_CACHE, which those libraries read at import time to
compute their cache paths. Without it, raw downloaded blobs (parquet shards,
model weights) land in ~/.cache/huggingface instead of this project, even
though cache_dir= is passed to from_pretrained / load_dataset /
hf_hub_download. (Deliberately not HF_HOME - that would also relocate the
`hf auth login` token away from where it is already stored.)
"""

import os
from pathlib import Path

# The repository root: common/ sits directly under it.
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
HF_HUB_CACHE_DIR = BASE_DIR / "hf_hub_cache"

# Results are filed per machine, since several boxes feed this repo and a run
# is only comparable if you know which one produced it. Each script writes to
# its own <OUTPUT_ROOT>/<script>_output/. Override when running elsewhere:
#   BENCH_OUTPUT_ROOT=output_SR630_6740_L4 python -m vit.vit_benchmark
OUTPUT_ROOT = Path(os.environ.get("BENCH_OUTPUT_ROOT",
                                  BASE_DIR / "output_SR650a_6787P_RTX6000"))

os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))
