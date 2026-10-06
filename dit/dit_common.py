"""What both DiT harnesses share: the model list and the prompt set.

dit_benchmark.py (offline latency / throughput) and server_dit_benchmark.py
(serving capacity) must agree on these, or their numbers stop being
comparable. Nothing here imports torch or diffusers.
"""

from common.paths import DATASET_DIR

MODELS = [
    "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
    # A UNet, not a DiT - kept as the pre-DiT reference point. Its denoiser is
    # pipe.unet rather than pipe.transformer; see denoiser() below.
    "stabilityai/stable-diffusion-xl-base-1.0",
]

# Models that need something beyond `pip install -r requirements.txt` before
# they will load. Checked up front so the failure is actionable instead of a
# stack trace from deep inside diffusers.
GATED = {
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large",
}

# Denoising steps when --steps is not given. Sana-Sprint is step-distilled: it
# is trained for 1-4 steps, so running it at the 20 the other models use would
# measure a configuration nobody deploys. It runs at 4, the top of that range.
DEFAULT_STEPS = 20
MODEL_STEPS = {
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers": 4,
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers": 4,
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


def steps_for(model, requested=None):
    """--steps if it was given, else the model's own default."""
    return requested if requested is not None else MODEL_STEPS.get(model, DEFAULT_STEPS)


def step_kwargs(model, steps):
    """Extra pipeline-call kwargs a model needs at this step count.

    SanaSprintPipeline defaults intermediate_timesteps=1.3, which is only
    defined for its 2-step schedule; at any other count it raises

        ValueError: Intermediate timesteps for SCM is not supported when
        num_inference_steps != 2.

    so every other count, the default of 4 included, has to clear it.
    """
    if model in MODEL_STEPS and steps != 2:
        return {"intermediate_timesteps": None}
    return {}


def denoiser(pipe):
    """(attribute name, module) of a pipeline's denoising network.

    The DiT pipelines (Sana, Sana-Sprint, PixArt, SD3.5) call it `transformer`; UNet
    pipelines (SDXL) call it `unet`. --compile and --quant target this module
    and nothing else, so the attribute has to follow the architecture rather
    than assume a transformer.
    """
    for name in ("transformer", "unet"):
        module = getattr(pipe, name, None)
        if module is not None:
            return name, module
    raise RuntimeError(f"{type(pipe).__name__} has neither a transformer nor a unet")


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
