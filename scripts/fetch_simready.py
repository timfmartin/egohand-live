"""Download the SimReady assets used by isaac/sharpa_hands.py.

Fetches only the needed folders, at pinned revisions, from two Hugging Face datasets
(both CC-BY-4.0) into ``third_party/simready/<org>/<asset>/``:

- the desk props (``--scene simready``) from nvidia/simready-assets (~360 MB);
- the Franka FR3 Duo (``--rig franka``) from frankarobotics/simready-assets (~14 MB).

Standard library only; skips files that are already present with the right size.

    python scripts/fetch_simready.py [--root third_party/simready] [--only props|franka]
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "isaac"))
from scenes import SIMREADY_DESK, SIMREADY_REPO, SIMREADY_REV, default_simready_root  # noqa: E402

FRANKA_REPO = "frankarobotics/simready-assets"
FRANKA_REV = "3d74c388b5d3dd5581ed4f4ff6f77fd8a4558bcf"
FRANKA_ASSETS = ["fr3_duo"]

API = "https://huggingface.co/api/datasets/{repo}/tree/{rev}/{path}?recursive=true"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
HEADERS = {"User-Agent": "egohand-live-fetch"}


def _get(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60)


def list_files(repo: str, rev: str, asset: str) -> list[dict]:
    url = API.format(repo=repo, rev=rev, path=urllib.parse.quote(asset))
    files = []
    while url:
        with _get(url) as r:
            files += [e for e in json.load(r) if e.get("type") == "file"]
            nxt = [p for p in r.headers.get("Link", "").split(",") if 'rel="next"' in p]
            url = nxt[0].split(";")[0].strip(" <>") if nxt else None
    return files


def fetch(repo: str, rev: str, assets: list[str], dest_root: Path) -> int:
    total = 0
    for k, asset in enumerate(assets, 1):
        files = list_files(repo, rev, asset)
        for f in files:
            dest = dest_root / f["path"]
            if dest.is_file() and dest.stat().st_size == f.get("size"):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = RESOLVE.format(repo=repo, rev=rev, path=urllib.parse.quote(f["path"]))
            tmp = dest.with_suffix(dest.suffix + ".part")
            with _get(url) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
            tmp.replace(dest)
            total += f.get("size", 0)
        print(f"[{k}/{len(assets)}] {repo.split('/')[0]}/{asset}: {len(files)} files")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=default_simready_root(REPO))
    ap.add_argument("--only", choices=["props", "franka"], help="fetch just one of the two sets")
    args = ap.parse_args()
    total = 0
    if args.only in (None, "props"):
        total += fetch(SIMREADY_REPO, SIMREADY_REV, sorted({p.asset for p in SIMREADY_DESK}), args.root / "nvidia")
    if args.only in (None, "franka"):
        total += fetch(FRANKA_REPO, FRANKA_REV, FRANKA_ASSETS, args.root / "frankarobotics")
    print(f"downloaded {total / 1e6:.0f} MB into {args.root}")


if __name__ == "__main__":
    main()
