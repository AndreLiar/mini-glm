"""Reproducibly vendor the Stage-2 corpus (ADR 0007).

Downloads Pride and Prejudice (Project Gutenberg #1342), strips the Gutenberg header/footer so only
the public-domain work remains, normalizes line endings, writes it under data/corpus/, and records a
sha256 manifest. Re-running reproduces the same file (and the integrity test checks the hash).

    python scripts/vendor_corpus.py                 # download from Gutenberg
    python scripts/vendor_corpus.py /path/raw.txt   # use an already-downloaded raw file
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
OUT = Path("data/corpus/pride_and_prejudice.txt")
MANIFEST = Path("data/corpus/MANIFEST.json")
START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def strip_boilerplate(raw: str) -> str:
    start = raw.index("\n", raw.index(START)) + 1
    end = raw.index(END)
    return raw[start:end].replace("\r\n", "\n").strip() + "\n"


def main() -> None:
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
    text = strip_boilerplate(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    MANIFEST.write_text(
        json.dumps(
            {"source_url": URL, "gutenberg_id": 1342, "title": "Pride and Prejudice",
             "sha256": sha, "chars": len(text)},
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT} ({len(text):,} chars)\nsha256 {sha}")


if __name__ == "__main__":
    main()
