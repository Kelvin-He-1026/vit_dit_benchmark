"""What all ViT harnesses share: the model list and the dataset.

vit_benchmark.py, server_vit_benchmark.py and diag_vit_inference.py must
agree on these, or their numbers stop being comparable.
"""

from pathlib import Path

MODELS = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "facebook/dinov2-giant",
]

DATASET_NAME = "ILSVRC/imagenet-1k"


def load_validation(cache_dir, log=print):
    """ImageNet-1k validation split, from the Hub or from the local cache.

    The normal path asks datasets for the split. That call needs the Hub even
    when every byte is already cached: the config id it looks up is a hash of
    the RESOLVED file list, so without Hub access it computes a different id
    and reports the cache as missing -

        ValueError: Couldn't find cache for ILSVRC/imagenet-1k for config
        'default-f380226377c594ae'

    which is what an expired `hf auth login` token looks like (ImageNet is a
    gated repo, so anonymous resolution fails too). The prepared Arrow shards
    next to that config are complete and readable on their own, so fall back
    to them rather than making a 6 GB re-download the price of a stale token.
    """
    from datasets import Dataset, concatenate_datasets, load_dataset

    try:
        return load_dataset(
            DATASET_NAME,
            data_files={"validation": "data/validation-*"},
            split="validation",
            cache_dir=str(cache_dir),
            verification_mode="no_checks",
        )
    except Exception as exc:  # noqa: BLE001 - any Hub/auth/cache failure
        shards = sorted(Path(cache_dir).glob(
            f"{DATASET_NAME.replace('/', '___')}/*/*/*/*-validation-*.arrow"))
        if not shards:
            raise RuntimeError(
                f"could not load {DATASET_NAME} ({type(exc).__name__}: "
                f"{str(exc)[:200]}), and no prepared shards were found under "
                f"{cache_dir}. Accept the terms at "
                f"https://huggingface.co/datasets/{DATASET_NAME} and run "
                f"`hf auth login`, then rerun."
            ) from exc
        log(f"Dataset    : Hub unavailable ({type(exc).__name__}); using the "
            f"{len(shards)} prepared shards cached in {cache_dir}")
        return concatenate_datasets([Dataset.from_file(str(p)) for p in shards])
