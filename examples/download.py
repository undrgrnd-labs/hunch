"""Download the datasets (and the EmbeddingGemma 2 checkpoint) the examples use. Asks before each download.

    uv run download.py                 # list what's available, what's here, and what each example needs
    uv run download.py sst5 injecagent # download these (asks first; --yes to skip the question)
    uv run download.py all

Datasets go to $EG2_DATA (default examples/data/hf). The model goes to the Hugging Face cache ($HF_HOME, default
~/.cache/huggingface), where eg2cpu.Model() looks for it. $EG2_MODEL_DIR is for a checkpoint kept elsewhere: eg2cpu
reads it from there instead, and this script doesn't write to it. Licences differ per dataset; read them before using
the data for anything else. Without a terminal to answer in, pass --yes.
"""

import argparse
import os
import sys
import urllib.request

from hunch_examples import HF

# Every download is pinned: the shipped hashes and labels refer to these revisions' contents.
REVISION = {
    "embeddinggemma-2": "914f7f89142e33e77833254d9c9b90c3cef7303b",
    "real-pi-coding-agent-traces-sessions": "8c593252ddad7dca08a0afc07896195fa73f2d6e",
    "tau2-bench-trajectories": "07264f1d253a618fc20966670eddc44246212916",
    "sst5": "e51bdcd8cd3a30da231967c1a249ba59361279a3",
    "InjecAgent": "623f1bf3ad8ed35abe71f9f9d8fd9d99ad65aeea",
    "synthetic_pii_finance_multilingual": "7b844d16738527a04264f50214cb426a4cea0897",
}
INJECAGENT_URL = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/{rev}/data/{name}.json"

# name -> (Hub repo or None, files / patterns, approximate size, licence)
DATASETS = {
    "embeddinggemma-2": ("model:google/embeddinggemma-2", ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
                         "1.4 GB", "Apache-2.0 (not gated)"),
    "real-pi-coding-agent-traces-sessions": ("MaxDevv/real-pi-coding-agent-traces-sessions", ["*.jsonl", "README.md"], "750 MB",
                                             "'other': a compilation of 21 sources, per-session licence in manifest.jsonl"),
    "tau2-bench-trajectories": ("AgentSuite/tau2-bench-trajectories", ["DeepSeek-*.jsonl", "README.md"], "37 MB (4 of 31 files)",
                                "none stated"),
    "sst5": ("SetFit/sst5", ["*.jsonl", "README.md"], "1.8 MB", "see the dataset card (Stanford Sentiment Treebank)"),
    "InjecAgent": (None, ["test_cases_dh_base", "test_cases_dh_enhanced", "test_cases_ds_base", "test_cases_ds_enhanced"],
                   "2.9 MB", "MIT (github.com/uiuc-kang-lab/InjecAgent)"),
    "synthetic_pii_finance_multilingual": ("gretelai/synthetic_pii_finance_multilingual", ["data/English_test-*.parquet", "README.md"],
                                           "2.7 MB", "Apache-2.0"),
}
OWNER = {"InjecAgent": "uiuc-kang-lab", "embeddinggemma-2": "google"}


def folder(name: str):
    repo = DATASETS[name][0]
    owner = OWNER.get(name) or repo.split("/")[0]
    return HF / f"{owner}__{name}"


def present(name: str) -> bool:
    if name == "embeddinggemma-2":
        try:
            import eg2cpu

            eg2cpu.Model()
            return True
        except Exception:
            return False
    d = folder(name)
    return d.is_dir() and any(d.iterdir())


def download(name: str) -> None:
    repo, files, *_ = DATASETS[name]
    if repo is None:  # InjecAgent: four files from GitHub
        d = folder(name)
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            url = INJECAGENT_URL.format(rev=REVISION[name], name=f)
            print(f"  {url}")
            urllib.request.urlretrieve(url, d / f"{f}.json")
        return
    from huggingface_hub import snapshot_download

    if repo.startswith("model:"):
        print("  ->", snapshot_download(repo[6:], revision=REVISION[name], allow_patterns=files))
        return
    print("  ->", snapshot_download(repo, repo_type="dataset", revision=REVISION[name], local_dir=folder(name),
                                   allow_patterns=files))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"any of: {', '.join(DATASETS)}, or 'all'")
    ap.add_argument("--yes", action="store_true", help="don't ask before each download")
    args = ap.parse_args()
    if not sys.stdout.isatty():  # one progress bar per file floods a log
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if not args.names:
        print(f"data dir: {HF}\n")
        for name, (repo, _, size, lic) in DATASETS.items():
            print(f"  [{'x' if present(name) else ' '}] {name:<38} {size:<22} licence: {lic}")
        print("\nuv run download.py <name> ... | all")
        return
    names = list(DATASETS) if args.names == ["all"] else args.names
    for name in names:
        match = [n for n in DATASETS if n.lower() == name.lower()]
        if not match:
            sys.exit(f"unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
        name = match[0]
        repo, _, size, lic = DATASETS[name]
        if present(name):
            print(f"{name}: already here")
            continue
        src = repo[6:] if repo and repo.startswith("model:") else repo or "github.com/uiuc-kang-lab/InjecAgent"
        if not args.yes:
            try:
                ok = input(f"Download {name} ({size}) from {src}? Licence: {lic}. [y/N] ").strip().lower() == "y"
            except EOFError:
                ok = False
            if not ok:
                print("  skipped" + ("" if sys.stdin.isatty() else " (no terminal to answer in: pass --yes)"))
                continue
        download(name)


if __name__ == "__main__":
    main()
