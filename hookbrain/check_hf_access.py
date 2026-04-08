import os
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


def check(repo_id: str, token: str) -> bool:
    api = HfApi()
    try:
        api.model_info(repo_id, token=token)
        print(f"✅ access ok: {repo_id}")
        return True
    except HfHubHTTPError as exc:
        print(f"❌ access failed: {repo_id}")
        print(f"   {exc}")
        return False


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not set")
        sys.exit(2)

    repos = [
        "facebook/tribev2",
        "meta-llama/Llama-3.2-3B",
        "unsloth/Llama-3.2-3B",
    ]
    ok = all(check(r, token) for r in repos)
    sys.exit(0 if ok else 1)
