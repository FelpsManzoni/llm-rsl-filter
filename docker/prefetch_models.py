from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def _collect_models() -> list[str]:
    model_vars = ["MODEL_1", "MODEL_2", "MODEL_1_FALLBACK", "MODEL_2_FALLBACK"]
    models: list[str] = []
    for var in model_vars:
        value = (os.getenv(var) or "").strip()
        if value and value not in models:
            models.append(value)
    return models


def _safe_model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def _looks_like_path(value: str) -> bool:
    return value.startswith("/") or value.startswith(".")


def main() -> int:
    model_cache_dir = (os.getenv("MODEL_CACHE_DIR") or "/models-cache").strip()
    model_store_dir = (os.getenv("MODEL_STORE_DIR") or "/models").strip()
    token = (os.getenv("HF_TOKEN") or "").strip() or None

    models = _collect_models()
    if not models:
        print("No models provided for prefetch; skipping.")
        return 0

    Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(model_store_dir).mkdir(parents=True, exist_ok=True)

    print(f"Prefetching {len(models)} model(s) into {model_store_dir}.")
    for model_id in models:
        if _looks_like_path(model_id):
            print(f"Skipping local model path: {model_id}")
            continue

        local_dir = Path(model_store_dir) / _safe_model_dir_name(model_id)
        print(f"Prefetching model: {model_id}")
        print(f"- Local directory: {local_dir}")
        snapshot_download(
            repo_id=model_id,
            cache_dir=model_cache_dir,
            local_dir=str(local_dir),
            token=token,
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        (local_dir / ".source_model").write_text(model_id + "\n", encoding="utf-8")

    print("Model prefetch completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
