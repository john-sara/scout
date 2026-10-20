"""Download the FastAPI docs (English markdown) at a pinned commit into data/fastapi_docs/.

The docs are MIT licensed (https://github.com/fastapi/fastapi). Pinning the commit keeps the
evaluation reproducible: the same questions are always checked against the same pages.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        run("git", "init", "-q", str(tmp))
        run("git", "remote", "add", "origin", REPO, cwd=tmp)
        run("git", "sparse-checkout", "set", DOCS_SUBDIR, cwd=tmp)
        run("git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT, cwd=tmp)
        run("git", "checkout", "-q", "FETCH_HEAD", cwd=tmp)
        src = tmp / DOCS_SUBDIR
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
        print(f"Copied {count} pages to {DEST}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
