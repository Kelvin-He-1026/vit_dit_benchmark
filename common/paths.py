"""Where everything lives on disk, for every benchmark.

Import this BEFORE torch, transformers, diffusers, datasets or
huggingface_hub. It sets HF_HUB_CACHE, which huggingface_hub reads at import
time to compute cache paths; set later, raw downloaded blobs (parquet shards,
model weights) land in ~/.cache/huggingface instead of this project, even
though cache_dir= is passed to every from_pretrained / load_dataset call.
(Deliberately not HF_HOME - that would also move the `hf auth login` token
away from where it is already stored.)

Models, datasets and the HF cache live at the repo root, shared by both
families; each script files its results under its own subfolder of
OUTPUT_ROOT.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
MODELS_DIR = REPO_ROOT / "models"
HF_HUB_CACHE_DIR = REPO_ROOT / "hf_hub_cache"



def _output_root():
    """BENCH_OUTPUT_ROOT if set, else output_<server>_<cpu>_<gpu> for this box.

    Results are filed per machine, since several boxes feed this repo and a
    run is only comparable if you know which one produced it. The name comes
    from what the machine reports about itself (common/hostinfo.py), so a
    fresh clone files its results correctly with nothing to configure;
    BENCH_SERVER supplies the server name where the machine cannot.
    """
    override = os.environ.get("BENCH_OUTPUT_ROOT")
    if override:
        return Path(override)
    from common import hostinfo
    return REPO_ROOT / f"output_{hostinfo.output_tag()}"


OUTPUT_ROOT = _output_root()
# Handed down so replica processes and the scripts a sweep launches use the
# same tree without probing the hardware again.
os.environ["BENCH_OUTPUT_ROOT"] = str(OUTPUT_ROOT)

os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))


def ensure_dirs(*extra):
    """Create the shared directories plus any script-specific ones."""
    for d in (DATASET_DIR, MODELS_DIR, HF_HUB_CACHE_DIR, *extra):
        d.mkdir(parents=True, exist_ok=True)
