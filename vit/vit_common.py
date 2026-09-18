"""What all ViT harnesses share: the model list and the dataset.

vit_benchmark.py, server_vit_benchmark.py and diag_vit_inference.py must
agree on these, or their numbers stop being comparable.
"""

MODELS = [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "facebook/dinov2-giant",
]

DATASET_NAME = "ILSVRC/imagenet-1k"
