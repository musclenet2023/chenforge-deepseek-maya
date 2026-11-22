"""Download a Hugging Face GGUF repo into a folder, verify SHA256 sums, and (for the
6-shard IQ2_XXS layout) hardlink the Engram file as shard 6.

Usage:
    python download_hf_model.py <repo_id> <dest-dir> [--sha256 sums.json]

`--sha256` takes a JSON file mapping file name -> expected sha256 (optional but
recommended; the HF tree sizes are checked against the download by default).
"""
import hashlib
import json
import os
import pathlib
import sys

REPO_IQ2 = "Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF"
SHARD6 = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf"
ENGRAM = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    repo, dest = sys.argv[1], sys.argv[2]
    sums = None
    if "--sha256" in sys.argv:
        i = sys.argv.index("--sha256")
        sums = json.load(open(sys.argv[i + 1], encoding="utf-8"))

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    for attempt in range(5):
        try:
            p = snapshot_download(repo_id=repo, local_dir=dest, max_workers=4)
            print("downloaded to:", p, flush=True)
            break
        except Exception as e:
            print("attempt", attempt, "failed:", repr(e), flush=True)
    else:
        sys.exit(1)

    m = pathlib.Path(dest)
    fails = []
    if sums:
        for name, want in sums.items():
            path = m / name
            if not path.exists():
                fails.append((name, "missing"))
                continue
            got = sha256(path)
            print(name, "OK" if got == want else f"MISMATCH got={got}", flush=True)
            if got != want:
                fails.append((name, got))

    # 6-shard layout: link the Engram file as shard 6 (Windows hardlink, then symlink)
    engram, shard6 = m / ENGRAM, m / SHARD6
    if engram.exists() and not shard6.exists():
        try:
            os.link(engram, shard6)
            print(f"hardlinked {SHARD6} -> {ENGRAM}", flush=True)
        except OSError as e:
            print("hardlink failed, trying symlink:", e, flush=True)
            shard6.symlink_to(engram.name)

    if fails:
        print("FAILURES:", fails)
        sys.exit(1)
    print("ALL COMPLETE" if sums else "DOWNLOAD COMPLETE (no sums file: sizes not re-checked)")


if __name__ == "__main__":
    main()
