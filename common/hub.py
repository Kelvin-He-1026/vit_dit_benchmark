"""Loading models when the Hugging Face Hub is unreachable.

Everything these benchmarks run is already in models/, but a from_pretrained
call still talks to the Hub to check for updates, and that lookup is what
fails when `hf auth login`'s token has expired: an INVALID token gets a 401
even on a public repo, where no token at all would have been fine -

    OSError: facebook/dinov2-giant is not a local folder and is not a valid
    model identifier listed on 'https://huggingface.co/models'

so a stale login looks exactly like a missing model. load_cached retries the
same call with local_files_only=True, which uses the cache and never touches
the network. If that fails too the model really is absent, and the original
Hub error is raised, since it says why the Hub was unusable.
"""


def load_cached(loader, *args, log=None, **kwargs):
    """Call `loader` (a from_pretrained), falling back to the local cache."""
    try:
        return loader(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any Hub/auth/network failure
        try:
            out = loader(*args, local_files_only=True, **kwargs)
        except Exception:  # noqa: BLE001 - not cached either; the Hub error is the useful one
            raise exc from None
        if log is not None:
            name = args[0] if args else kwargs.get("pretrained_model_name_or_path", "?")
            log(f"  {name}: Hub unavailable ({type(exc).__name__}); loaded from "
                f"the local cache. Run `hf auth login` to refresh the token.")
        return out
