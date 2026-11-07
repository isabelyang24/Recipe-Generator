"""Verify and extract this contribution's processed image archives."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    root, archives = HERE / "processed", HERE / "archives"
    index = json.loads((archives / "index.json").read_text())
    seen = set()
    for entry in index["archives"]:
        path = archives / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {path.name}")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) != entry["images"]:
                raise ValueError("Unexpected archive member count")
            for info in infos:
                target = (root / info.filename).resolve()
                if (not info.filename.startswith("images/") or
                        not target.is_relative_to((root / "images").resolve()) or
                        info.filename in seen):
                    raise ValueError(f"Unsafe or repeated archive member: {info.filename}")
                seen.add(info.filename)
                content = archive.read(info)
                if target.exists():
                    if target.read_bytes() != content:
                        raise FileExistsError(f"Refusing to replace modified image: {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != index["image_count"] or seen != {r["image_path"] for r in rows}:
        raise ValueError("Archive files do not match the manifest")
    for row in rows:
        path = root / row["image_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["processed_sha256"]:
            raise ValueError(f"Processed image checksum mismatch: {path}")
    print(f"Verified and extracted {len(rows)} processed images")


if __name__ == "__main__":
    main()
