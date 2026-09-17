"""Compare a PNG dataset with a separately downloaded course reference ZIP using SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def verify(root: Path, archive: Path) -> dict:
    differences, expected = [], set()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            path = Path(entry.filename)
            if not entry.filename.startswith("keras_png_slices_data/") or path.suffix != ".png":
                continue
            rel = path.relative_to("keras_png_slices_data")
            if ".." in rel.parts:
                raise ValueError("Invalid reference archive path")
            if str(rel) in expected:
                raise ValueError(f"Duplicate archive member: {rel}")
            expected.add(str(rel))
            local = root / rel
            if not local.is_file() or hashlib.sha256(local.read_bytes()).digest() != hashlib.sha256(z.read(entry)).digest():
                differences.append(str(rel))
    extras = sorted(str(p.relative_to(root)) for p in root.rglob("*.png") if str(p.relative_to(root)) not in expected)
    return {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "n_reference_pngs": len(expected), "missing_or_different": differences,
            "extra_pngs": extras, "matches": bool(expected) and not differences and not extras}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--reference_zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/results/archive_comparison.json"))
    args = parser.parse_args()
    report = verify(args.data_root, args.reference_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["matches"] else 1)


if __name__ == "__main__":
    main()
