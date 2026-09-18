"""What both DiT harnesses share: the model list and the prompt set.

dit_benchmark.py (offline latency / throughput) and server_dit_benchmark.py
(serving capacity) must agree on these, or their numbers stop being
comparable. Nothing here imports torch or diffusers.
"""

from common.paths import DATASET_DIR

MODELS = [
    "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
]

# Models that need something beyond `pip install -r requirements.txt` before
# they will load. Checked up front so the failure is actionable instead of a
# stack trace from deep inside diffusers.
GATED = {
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
}

UNSUPPORTED = {
    "OmniGen2/OmniGen2": (
        "OmniGen2's model_index.json declares _class_name=OmniGen2Pipeline, but that "
        "class does not exist in any released diffusers (checked 0.39.0 and 0.40.0) "
        "nor on diffusers main, and the model repo ships only a custom transformer "
        "and scheduler - no pipeline. Running it requires the upstream package from "
        "github.com/VectorSpaceLab/OmniGen2, which pins torch 2.6.0 and would "
        "conflict with this environment (torch 2.13). Install it in a separate venv "
        "and benchmark it there."
    ),
}


def load_prompts(n):
    """The first n COCO 2014 validation captions, downloaded once and cached."""
    from huggingface_hub import hf_hub_download

    prompt_file = hf_hub_download(
        repo_id="byliutao/coco2014val_10k",
        repo_type="dataset",
        filename="test.txt",
        cache_dir=str(DATASET_DIR),
    )
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[:n]
