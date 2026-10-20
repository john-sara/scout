"""Download the FastAPI docs (English markdown) at a pinned commit into data/fastapi_docs/.

The docs are MIT licensed (https://github.com/fastapi/fastapi). Pinning the commit keeps the
evaluation reproducible: the same questions are always checked against the same pages.
"""
import io
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

REPO = "https://github.com/fastapi/fastapi.git"
COMMIT = "03002a00b48335fdd8159edb7920157b508d8c15"
DOCS_SUBDIR = "docs/en/docs"
# Pages that are about the project itself, not about using FastAPI.
SKIP = {
    "_llm-test.md", "release-notes.md", "fastapi-people.md", "external-links.md",
    "newsletter.md", "translations.md", "translation-banner.md", "management.md",
}
DEST = Path(__file__).resolve().parent.parent / "data" / "fastapi_docs"


def run(*cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def copy_pages(src: Path) -> int:
    if DEST.exists():
        shutil.rmtree(DEST)
    count = 0
    for p in sorted(src.rglob("*.md")):
        rel = p.relative_to(src)
        if rel.as_posix() in SKIP:
            continue
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, out)
        count += 1
    return count


def fetch_with_zip() -> int:
    """Fallback for machines without git: download the pinned commit as a zip from GitHub."""
    print("git not available, downloading the docs as a zip instead (about 30 MB)...")
    r = requests.get(f"https://codeload.github.com/fastapi/fastapi/zip/{COMMIT}", timeout=120)
    r.raise_for_status()
    tmp = Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            members = [m for m in z.namelist() if f"/{DOCS_SUBDIR}/" in m and m.endswith(".md")]
            for m in members:
                target = tmp / m.split(f"/{DOCS_SUBDIR}/", 1)[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(m))
        return copy_pages(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if shutil.which("git") is None:
        print(f"Copied {fetch_with_zip()} pages to {DEST}")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        run("git", "init", "-q", str(tmp))
        run("git", "remote", "add", "origin", REPO, cwd=tmp)
        run("git", "sparse-checkout", "set", DOCS_SUBDIR, cwd=tmp)
        run("git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT, cwd=tmp)
        run("git", "checkout", "-q", "FETCH_HEAD", cwd=tmp)
        count = copy_pages(tmp / DOCS_SUBDIR)
        print(f"Copied {count} pages to {DEST}")
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"git download failed ({e}); trying the zip download instead.")
        print(f"Copied {fetch_with_zip()} pages to {DEST}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
