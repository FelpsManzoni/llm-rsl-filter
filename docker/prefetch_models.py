from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download


def _collect_models() -> list[str]:
    model_vars = ["MODEL_1", "MODEL_2", "MODEL_1_FALLBACK", "MODEL_2_FALLBACK"]
    models: list[str] = []
    for var in model_vars:
        value = (os.getenv(var) or "").strip()
        if value and value not in models:
            models.append(value)
    return models


def main() -> int:
    model_cache_dir = (os.getenv("MODEL_CACHE_DIR") or "/models-cache").strip()
    token = (os.getenv("HF_TOKEN") or "").strip() or None

    models = _collect_models()
    if not models:
        print("No models provided for prefetch; skipping.")
        return 0

    print(f"Prefetching {len(models)} model(s) into {model_cache_dir}.")
    for model_id in models:
        print(f"Prefetching model: {model_id}")
        snapshot_download(
            repo_id=model_id,
            cache_dir=model_cache_dir,
            token=token,
            resume_download=True,
            local_dir_use_symlinks=False,
        )

    print("Model prefetch completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
