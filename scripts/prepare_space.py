"""Assemble a Hugging Face Space directory, ready to push.

The Space needs a different root layout from this repo: Hugging Face reads
`README.md` and `requirements.txt` from the repo root and cannot be pointed
elsewhere, and the project root already uses both for the Docker/Render path.
Rather than overwrite them, this stages a self-contained copy.

It also writes a `.gitattributes` marking the model files for Git LFS. Without
it the push is rejected - `models/` is ~880 MB and individual files exceed
GitHub-style size limits.

Run:  python -m scripts.prepare_space
      python -m scripts.prepare_space --out C:\\tmp\\my-space --clean
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from src import config  # noqa: E402

# Everything the Space needs at runtime, and nothing else. The raw dataset,
# anomalib checkpoints, node_modules and the venv are all left behind.
# Hugging Face rejects a push containing binary files that are not in LFS, and
# that applies to the sample JPEGs too - not just the large model files. Size is
# not the trigger: the demo images are ~130 KB each and were still rejected.
GITATTRIBUTES = """\
*.pt filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.webp filter=lfs diff=lfs merge=lfs -text
*.bmp filter=lfs diff=lfs merge=lfs -text
"""


def copy_tree(src: Path, dst: Path, *, ignore: shutil.ignore_patterns | None = None) -> int:
    if not src.exists():
        return 0
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=config.PROJECT_ROOT / "build" / "hf-space",
        help="Directory to assemble into.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete --out first.")
    args = parser.parse_args()

    root = config.PROJECT_ROOT
    out: Path = args.out

    space_readme = root / "deploy" / "README_SPACE.md"
    space_reqs = root / "requirements-gradio.txt"
    for required in (space_readme, space_reqs, root / "deploy" / "gradio_app.py"):
        if not required.exists():
            print(f"missing: {required}")
            return 1

    if out.exists():
        if not args.clean:
            print(f"{out} already exists. Pass --clean to replace it.")
            return 1
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Root files, renamed to what Hugging Face expects.
    shutil.copyfile(space_readme, out / "README.md")
    shutil.copyfile(space_reqs, out / "requirements.txt")
    (out / ".gitattributes").write_text(GITATTRIBUTES)
    # Running the app from the staged directory creates bytecode caches; keep
    # them out of the Space repo.
    (out / ".gitignore").write_text("__pycache__/\n*.pyc\n.gradio/\n")

    # Application code. __pycache__ would otherwise bloat the push.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    copy_tree(root / "src", out / "src", ignore=ignore)

    (out / "deploy").mkdir(exist_ok=True)
    shutil.copyfile(root / "deploy" / "gradio_app.py", out / "deploy" / "gradio_app.py")
    init = root / "deploy" / "__init__.py"
    if init.exists():
        shutil.copyfile(init, out / "deploy" / "__init__.py")
    copy_tree(root / "deploy" / "samples", out / "deploy" / "samples")

    # Trained artefacts: memory banks and classifiers.
    copy_tree(root / "models", out / "models")

    # Only metrics.json is needed at serve time; splits/feature caches/anomalib
    # checkpoints are training by-products.
    for category in config.trained_categories():
        metrics = config.metrics_path(category)
        if metrics.exists():
            target = out / "outputs" / category
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(metrics, target / "metrics.json")

    total = 0
    print(f"{'item':<24}{'files':>8}{'size MB':>12}")
    print("-" * 44)
    for name in ("src", "deploy", "models", "outputs"):
        path = out / name
        if not path.exists():
            continue
        files = [p for p in path.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files) / 1024 / 1024
        total += size
        print(f"{name:<24}{len(files):>8}{size:>12.1f}")
    roots = [p for p in out.glob("*") if p.is_file()]
    root_size = sum(p.stat().st_size for p in roots) / 1024 / 1024
    total += root_size
    print(f"{'(root files)':<24}{len(roots):>8}{root_size:>12.2f}")
    print("-" * 44)
    print(f"{'TOTAL':<24}{'':>8}{total:>12.1f}")

    print(f"\nassembled -> {out}\n")
    print("Next:")
    print("  1. Create a Space at https://huggingface.co/new-space")
    print("     SDK = Gradio, hardware = CPU basic (free)")
    print("  2. Set the secret: Settings -> Variables and secrets -> New secret")
    print("     DEFECT_DETECTOR_API_KEY = <a long random string>")
    print(f"  3. cd {out}")
    print("     git init && git lfs install")
    print("     git remote add origin https://huggingface.co/spaces/<user>/<space>")
    print("     git add . && git commit -m 'Deploy defect detector'")
    print("     git push -u origin main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
