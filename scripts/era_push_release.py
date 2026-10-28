"""Push data/release/<version> to Hugging Face and/or Zenodo when tokens exist.

  uv run python scripts/era_push_release.py --version v0.1-era
  uv run python scripts/era_push_release.py --version v0.1-era --hf-only
  uv run python scripts/era_push_release.py --version v0.1-era --zenodo-only

Env: HF_TOKEN, ZENODO_TOKEN (optional .env). Missing token → skip that target, exit 0 if at least
local package exists (local-complete). Exit 2 if neither token and --require-remote.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("era_push")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_REPO = "ANONYMIZED/CulturePersona-Seoul"  # real org/name after acceptance


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def push_hf(rel: Path, repo_id: str, *, public: bool = False) -> str | None:
    """Upload to HF. **The default is a private repo** (ADR-012).

    The previous version created a public dataset with `private=False`, and the
    `era_b2_after_b1.sh` watcher called this function with no visibility choice.
    Public upload happens only with an explicit `--public` flag.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.warning("HF_TOKEN unset — skip Hugging Face")
        return None
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    log.info("HF repo %s — visibility=%s", repo_id, "public" if public else "private")
    create_repo(
        repo_id=repo_id, repo_type="dataset", exist_ok=True, private=not public, token=token
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(rel),
        commit_message=f"sfac_persona era release {rel.name}",
        ignore_patterns=["*.zip", "code_snapshot/**"],
        token=token,
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    log.info("HF OK %s", url)
    return url


def push_zenodo(rel: Path, *, public: bool = False) -> str | None:
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        log.warning("ZENODO_TOKEN unset — skip Zenodo")
        return None
    # Minimal deposition upload via requests (Zenodo REST).
    import json
    import zipfile

    import requests

    zpath = rel.parent / f"{rel.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in rel.rglob("*"):
            if p.is_file() and "code_snapshot" not in p.parts:
                zf.write(p, p.relative_to(rel).as_posix())
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(
        "https://zenodo.org/api/deposit/depositions",
        params={"access_token": token},
        json={},
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    r.raise_for_status()
    dep = r.json()
    bucket = dep["links"]["bucket"]
    with open(zpath, "rb") as fp:
        requests.put(
            f"{bucket}/{zpath.name}",
            params={"access_token": token},
            data=fp,
            timeout=600,
        ).raise_for_status()
    meta = {
        "metadata": {
            "title": f"sfac_persona era synthetic personas ({rel.name})",
            "upload_type": "dataset",
            "description": "Survey-grounded synthetic personas (Seoul cultural participation panel estimates). Aggregate-table sampler only.",
            "creators": [{"name": "ANONYMIZED"}],  # real name after acceptance
            "access_right": "open" if public else "restricted",
            "license": "CC-BY-4.0",
        }
    }
    requests.put(
        dep["links"]["self"],
        params={"access_token": token},
        data=json.dumps(meta),
        headers={**headers, "Content-Type": "application/json"},
        timeout=120,
    ).raise_for_status()
    url = dep["links"].get("html", dep["links"]["self"])
    log.info("Zenodo deposition drafted (publish manually if needed): %s", url)
    return url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0.1-era")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--hf-only", action="store_true")
    ap.add_argument("--zenodo-only", action="store_true")
    ap.add_argument("--require-remote", action="store_true")
    ap.add_argument(
        "--public",
        action="store_true",
        help="Upload as public. Default is private — stops an automatic watcher from publishing (ADR-012).",
    )
    args = ap.parse_args()
    load_env()
    rel = ROOT / "data" / "release" / args.version
    if not (rel / "MANIFEST.sha256").exists():
        raise SystemExit(f"missing local package {rel} — run era_package_release.py first")

    urls = []
    do_hf = not args.zenodo_only
    do_zen = not args.hf_only
    if do_hf:
        u = push_hf(rel, args.hf_repo, public=args.public)
        if u:
            urls.append(u)
    if do_zen:
        u = push_zenodo(rel, public=args.public)
        if u:
            urls.append(u)

    note = ROOT / "outputs" / "reports" / "era_release_push.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    if urls:
        note.write_text(
            "# Era release remote\n\n" + "\n".join(f"- {u}" for u in urls) + "\n", encoding="utf-8"
        )
        log.info("wrote %s", note)
        return 0
    note.write_text(
        "# Era release remote\n\n- local package only (`data/release/"
        + args.version
        + "`). HF_TOKEN/ZENODO_TOKEN unset — remote deferred.\n",
        encoding="utf-8",
    )
    log.info("local-complete; remote deferred → %s", note)
    return 2 if args.require_remote else 0


if __name__ == "__main__":
    raise SystemExit(main())
