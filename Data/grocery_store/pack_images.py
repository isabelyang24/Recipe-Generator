"""Package prepared images in small, checksum-indexed archives for GitHub."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    root, destination = HERE / "processed", HERE / "archives"
    destination.mkdir(exist_ok=True)
    if any(destination.glob("*.zip")):
        raise FileExistsError("Archives exist; move them before rebuilding")
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    entries = []
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        for i in range(0, len(subset), 250):
            name = f"{split}-{i // 250 + 1:03}.zip"
            target = destination / name
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as archive:
                for row in subset[i:i + 250]:
                    # Fixed timestamp makes repeated packaging byte-for-byte stable.
                    info = zipfile.ZipInfo(row["image_path"], date_time=(2026, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, (root / row["image_path"]).read_bytes())
            entries.append({"file": name, "images": len(subset[i:i + 250]),
                            "bytes": target.stat().st_size,
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination / "index.json").write_text(json.dumps(
        {"archives": entries, "image_count": len(rows)}, indent=2) + "\n")
    print(f"Packaged {len(rows)} images in {len(entries)} archives")


if __name__ == "__main__":
    main()
