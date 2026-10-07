"""Build the downloadable release assets that are too large or too detailed for Git.

The repository tracks curated aggregate evidence under ``results/``. Model
weights and complete per-record evaluation, policy and audit bundles are
published as GitHub Release assets instead (see docs/ARTIFACTS.md). This script
collects them from a local checkout that still holds the original outputs and
writes them, plus ``SHA256SUMS`` and ``release-manifest.json``, to one folder
ready for upload.

Bundles are archived with their original repository-relative paths, so
extracting one from the repository root restores the layout the reproduction
scripts expect, e.g. ``artifacts/comparison/runpod-s42/...``. Their contents
are frozen originals and are copied byte for byte; logs and manifests inside
them can record the absolute paths of the machines that produced them.

Usage (from the repository root):
    python scripts/build_release_bundles.py --source-root /path/to/checkout-with-artifacts
    gh release create results-s42-v1 release/results-s42-v1/* --title "Seed-42 results" \
        --notes-file release/results-s42-v1/RELEASE_NOTES.md
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io_utils import sha256_file  # noqa: E402

TAG = "results-s42-v1"
# Commit that last tracked the original smoke exports. It exists only in the
# pre-publication history (archived privately), so the public repository skips
# rebuilding this asset; the published copy is the frozen original.
SMOKE_COMMIT = "4c6159d"
SMOKE_PATHS = ("artifacts/evaluation/smoke-evaluator", "artifacts/evaluation/smoke-evaluator-v2")

PTBXL_ATTRIBUTION = (
    "PTB-XL attribution: Wagner P, Strodthoff N, Bousseljot R-D, Samek W, Schaeffter T. PTB-XL, "
    "a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet, 2022. "
    "https://doi.org/10.13026/kfzx-aw45, and Wagner et al., Scientific Data 7:154 (2020), "
    "https://doi.org/10.1038/s41597-020-0495-6. Licensed under CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/). These assets are derived from PTB-XL: "
    "models were trained on its waveforms, and exports contain its labels, pseudonymous patient "
    "identifiers and metadata together with model predictions. PTB-XL's authors do not endorse "
    "this project."
)

WEIGHTS = {
    "fixed12-s42.best.pt": "runpod-results/checkpoints/fixed12-s42/best.pt",
    "fixed2-s42.best.pt": "runpod-results/checkpoints/fixed2-s42/best.pt",
    "random2-s42.best.pt": "runpod-results/checkpoints/random2-s42/best.pt",
    "historical-random12-s42.best.pt": "checkpoints/best.pt",
}
BUNDLES = {
    "runpod-s42-comparison-full.tar.gz": (
        "Complete seed-42 comparison: per-record validation/test exports, frozen policies, "
        "audits, logs and report inputs.",
        ["artifacts/comparison/runpod-s42"],
    ),
    "runpod-s42-relaxed-risk-full.tar.gz": (
        "Complete AFIB-or-AFLT relaxed-risk sensitivity: protocols, validation policies, "
        "test audits and logs (including the retired 5% scenarios).",
        ["artifacts/comparison/runpod-s42-relaxed-risk"],
    ),
    "historical-model-full.tar.gz": (
        "Historical-model evidence cited in the docs: validation and test exports, the final AFIB, "
        "all-label, use-contract v1 and legacy patient-level policies, and their audits.",
        [
            "artifacts/evaluation/20260902T072749899755Z-best-b7fb8247",
            "artifacts/evaluation/afib-locked-test-export-v1",
            "artifacts/selective/afib-validation-policy-platt",
            "artifacts/selective/afib-positive-exploratory-risk40-v2",
            "artifacts/audit/afib-locked-test-audit-v2",
            "artifacts/audit/afib-positive-exploratory-test-v2",
            "artifacts/selective-all/all-labels-baseline-v2",
            "artifacts/selective-all/use-contract-routed-v2",
            "artifacts/selective-all/patient-v2-verified",
            "artifacts/audit-all/all-labels-test-v3",
            "artifacts/audit-all/use-contract-routed-v2-audit",
            "artifacts/audit-all/patient-v2-final-audit",
        ],
    ),
}


def _reset(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def write_tar(source_root: Path, members: list[str], destination: Path) -> None:
    with destination.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for member in members:
                root = source_root / member
                if not root.is_dir():
                    raise FileNotFoundError(root)
                for path in sorted([root, *root.rglob("*")]):
                    if path.name == ".DS_Store":
                        continue
                    archive.add(path, arcname=path.relative_to(source_root).as_posix(),
                                recursive=False, filter=_reset)


def smoke_commit_available(repo_root: Path) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{SMOKE_COMMIT}^{{commit}}"],
        capture_output=True,
    ).returncode == 0


def write_smoke_bundle(repo_root: Path, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar", SMOKE_COMMIT, *SMOKE_PATHS],
        check=True, capture_output=True,
    ).stdout
    with destination.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(archive)
    # Fail early if git archive returned nothing useful.
    with tarfile.open(fileobj=io.BytesIO(archive)) as check:
        if not any(name.endswith("manifest.json") for name in check.getnames()):
            raise RuntimeError("smoke export archive is empty")


def release_notes(entries: list[dict]) -> str:
    lines = [
        f"# {TAG}",
        "",
        "Seed-42 model weights and complete per-record evidence bundles for the",
        "Reduced Leads project. Curated summaries are tracked in the repository",
        "under `results/`; see `docs/ARTIFACTS.md` for the tracking policy and",
        "`docs/EXPERIMENTS.md` for run names.",
        "",
        "All evaluation outputs are derived from PTB-XL 1.0.3 (CC BY 4.0) and",
        "contain its pseudonymous patient identifiers. They are exploratory research",
        "artifacts, not clinical tools. Verify downloads with `shasum -a 256 -c SHA256SUMS`.",
        "",
        f"{PTBXL_ATTRIBUTION}",
        "",
        "| Asset | Size | Description |",
        "|---|---:|---|",
    ]
    for entry in entries:
        size = f"{entry['bytes'] / 1e6:.1f} MB" if entry["bytes"] >= 1e6 else f"{entry['bytes'] / 1e3:.0f} KB"
        lines.append(f"| `{entry['name']}` | {size} | {entry['description']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=repo_root,
                        help="checkout that contains the local checkpoints/, runpod-results/ and artifacts/")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "release" / TAG)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for name, relative in WEIGHTS.items():
        shutil.copy2(source_root / relative, out_dir / name)
        entries.append({"name": name, "source": relative,
                        "description": f"Inference checkpoint (`best.pt`) for run `{name.split('.')[0]}`."})
    for name, (description, members) in BUNDLES.items():
        print(f"[release] {name}")
        write_tar(source_root, members, out_dir / name)
        entries.append({"name": name, "source": members, "description": description})
    smoke = "legacy-smoke-exports.tar.gz"
    if smoke_commit_available(repo_root):
        write_smoke_bundle(repo_root, out_dir / smoke)
        entries.append({"name": smoke, "source": [f"{SMOKE_COMMIT}:{p}" for p in SMOKE_PATHS],
                        "description": "Frozen original smoke exports formerly tracked in Git (real PTB-XL rows); "
                                       "replaced in the repository by examples/synthetic-evaluation."})
    else:
        print(f"[release] skipping {smoke}: commit {SMOKE_COMMIT} is not in this history; "
              "reuse the asset from the existing release")

    for entry in entries:
        path = out_dir / entry["name"]
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    (out_dir / "SHA256SUMS").write_text("".join(f"{e['sha256']}  {e['name']}\n" for e in entries))
    (out_dir / "release-manifest.json").write_text(json.dumps({"tag": TAG, "assets": entries}, indent=2) + "\n")
    (out_dir / "RELEASE_NOTES.md").write_text(release_notes(entries))
    print(f"[release] wrote {len(entries)} assets to {out_dir}")


if __name__ == "__main__":
    main()
