"""Audit real PNG identities, mask levels and case-level split leakage before training."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part4_recognition.oasis_data import (  # noqa: E402
    OASISDataset, find_oasis_root, grey_levels_to_labels, parse_case_and_slice,
)


def audit(root: Path) -> dict:
    report = {"root": str(root), "splits": {}, "case_overlap": {}}
    cases = {}
    for split in ("train", "validate", "test"):
        files = sorted((root / f"keras_png_slices_{split}").glob("*.png"))
        if not files:
            raise ValueError(f"Empty split: {split}")
        keys = [parse_case_and_slice(p.name) for p in files]
        if (-1, -1) in keys or len(set(keys)) != len(keys):
            raise ValueError(f"Invalid or duplicate case/slice identities: {split}")
        masks = OASISDataset._resolve_masks(files, root / f"keras_png_slices_seg_{split}")
        cases[split] = {key[0] for key in keys}
        counts = np.zeros(4, dtype=np.int64)
        levels, shapes = set(), set()
        for image_path, mask_path in zip(files, masks):
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError(f"Shape mismatch: {image_path}, {mask_path}")
                image.load()
                raw = np.asarray(mask.convert("L"))
                shapes.add(image.size)
                levels.update(np.unique(raw).tolist())
                counts += np.bincount(grey_levels_to_labels(raw).ravel(), minlength=4)
        report["splits"][split] = {"n_slices": len(files), "n_cases": len(cases[split]),
            "case_ids": sorted(cases[split]), "shapes": sorted(shapes), "mask_grey_levels": sorted(levels),
            "pixel_counts": counts.tolist(), "pixel_fractions": (counts / counts.sum()).tolist()}
    for a, b in combinations(cases, 2):
        overlap = sorted(cases[a] & cases[b])
        report["case_overlap"][f"{a}_{b}"] = overlap
        if overlap:
            raise ValueError(f"Patient/case leakage across {a}/{b}: {overlap}")
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/results/oasis_audit.json"))
    args = parser.parse_args()
    report = audit(find_oasis_root(args.data_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
