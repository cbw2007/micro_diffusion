import json
import os
import math
from argparse import ArgumentParser
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from datasets import load_dataset
from streaming.base import MDSWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
Example usage:
python convert.py --local_mds_dir ./textcaps/mds/
"""


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--local_mds_dir",
        type=str,
        help="Directory to store mds shards.",
    )
    parser.add_argument(
        "--save_text_bboxes",
        default=False,
        action="store_true",
        help="If True, save OCR/text bounding boxes as a JSON string.",
    )
    parser.add_argument(
        "--require_text_bboxes",
        default=False,
        action="store_true",
        help="If True, fail when no supported bbox field is present.",
    )
    parser.add_argument(
        "--bbox_format",
        type=str,
        default="auto",
        choices=("auto", "xyxy", "xywh"),
        help="Format for list-like bbox fields. Dict fields with named corners/sizes are inferred.",
    )
    args = parser.parse_args()
    return args


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    keys = batch[0].keys()
    data = {k: [] for k in keys}
    for b in batch:
        for k, v in b.items():
            data[k].append(v)
    return data


def _first_present_key(sample: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in sample:
            return key
    return None


def _to_float_list(value: Any) -> List[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return []
    if not isinstance(value, (list, tuple)):
        return []
    return [float(v) for v in value]


def _normalize_bbox(
    raw_bbox: Any,
    width: int,
    height: int,
    bbox_format: str = "auto",
) -> Optional[List[float]]:
    inferred_format = bbox_format
    if isinstance(raw_bbox, dict):
        if all(k in raw_bbox for k in ("x0", "y0", "x1", "y1")):
            raw_bbox = [
                raw_bbox["x0"],
                raw_bbox["y0"],
                raw_bbox["x1"],
                raw_bbox["y1"],
            ]
            inferred_format = "xyxy"
        elif all(k in raw_bbox for k in ("x", "y", "w", "h")):
            raw_bbox = [
                raw_bbox["x"],
                raw_bbox["y"],
                raw_bbox["w"],
                raw_bbox["h"],
            ]
            inferred_format = "xywh"
        elif all(k in raw_bbox for k in ("left", "top", "width", "height")):
            raw_bbox = [
                raw_bbox["left"],
                raw_bbox["top"],
                raw_bbox["width"],
                raw_bbox["height"],
            ]
            inferred_format = "xywh"
    bbox = _to_float_list(raw_bbox)
    if len(bbox) < 4:
        return None

    x0, y0, a, b = bbox[:4]
    if inferred_format == "xyxy":
        x1, y1 = a, b
    elif inferred_format == "xywh":
        x1, y1 = x0 + max(a, 0.0), y0 + max(b, 0.0)
    else:
        # Prefer xyxy when the last corner is already larger than the first.
        if a > x0 and b > y0:
            x1, y1 = a, b
        else:
            x1, y1 = x0 + max(a, 0.0), y0 + max(b, 0.0)

    # Handle normalized coordinates.
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height

    x0 = max(0.0, min(float(width), x0))
    x1 = max(0.0, min(float(width), x1))
    y0 = max(0.0, min(float(height), y0))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def extract_text_bboxes(
    sample: Dict[str, Any],
    bbox_format: str = "auto",
) -> List[List[float]]:
    """Extract OCR/text bboxes from common TextCaps-style fields."""
    width = int(sample["image_width"])
    height = int(sample["image_height"])
    bbox_key = _first_present_key(
        sample,
        [
            "ocr_bboxes",
            "ocr_boxes",
            "ocr_bbox",
            "ocr_box",
            "bboxes",
            "boxes",
            "bbox",
        ],
    )
    if bbox_key is None:
        return []

    raw_bboxes = sample[bbox_key]
    if isinstance(raw_bboxes, dict):
        raw_bboxes = raw_bboxes.get("bboxes", raw_bboxes.get("boxes", []))
    if hasattr(raw_bboxes, "tolist"):
        raw_bboxes = raw_bboxes.tolist()
    if not isinstance(raw_bboxes, (list, tuple)):
        return []
    if len(raw_bboxes) >= 4 and all(
        not isinstance(v, (list, tuple, dict)) for v in raw_bboxes[:4]
    ):
        raw_bboxes = [raw_bboxes]

    bboxes = []
    for raw_bbox in raw_bboxes:
        bbox = _normalize_bbox(
            raw_bbox,
            width=width,
            height=height,
            bbox_format=bbox_format,
        )
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def main():
    args = parse_arguments()

    ds = load_dataset(
        "HuggingFaceM4/TextCaps",
        split="train+validation",
    )
    loader = DataLoader(
        ds,
        batch_size=512,
        collate_fn=collate_fn,
    )

    bbox_available = _first_present_key(ds[0], [
        "ocr_bboxes",
        "ocr_boxes",
        "ocr_bbox",
        "ocr_box",
        "bboxes",
        "boxes",
        "bbox",
    ])
    if args.save_text_bboxes:
        if bbox_available is None:
            msg = (
                "No supported TextCaps bbox field found. "
                f"Available fields: {list(ds[0].keys())}"
            )
            if args.require_text_bboxes:
                raise ValueError(msg)
            print(msg + " Writing empty text_bboxes lists.")
        else:
            print(f"Using TextCaps bbox field: {bbox_available}")

    keys = ["height", "width", "jpg", "image_id", "org_captions"]
    if args.save_text_bboxes:
        keys.append("text_bboxes")
    samples = {k: [] for k in keys}

    for i, batch in tqdm(enumerate(loader)):
        samples["height"].extend(batch["image_height"])
        samples["width"].extend(batch["image_width"])
        samples["jpg"].extend(batch["image"])
        samples["image_id"].extend(batch["image_id"])
        samples["org_captions"].extend(batch["reference_strs"])
        if args.save_text_bboxes:
            for sample_idx in range(len(batch["image"])):
                sample = {k: v[sample_idx] for k, v in batch.items()}
                samples["text_bboxes"].append(
                    extract_text_bboxes(sample, bbox_format=args.bbox_format)
                )

    print(f"Total {len(samples['jpg'])} samples in textcaps dataset")

    columns = {
        "height": "int32",
        "width": "int32",
        "jpg": "jpeg",
        "image_id": "str",
        "caption": "str",
    }
    if args.save_text_bboxes:
        columns["text_bboxes"] = "str"

    writer = MDSWriter(
        out=args.local_mds_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for i in range(len(samples["jpg"])):
        try:
            mds_sample = {
                "height": samples["height"][i],
                "width": samples["width"][i],
                "jpg": samples["jpg"][i],
                "image_id": samples["image_id"][i],
                "caption": samples["org_captions"][i][0],
            }
            if args.save_text_bboxes:
                mds_sample["text_bboxes"] = json.dumps(samples["text_bboxes"][i])
            writer.write(mds_sample)
        except Exception as e:
            print(
                f"Something went wrong in reading caption, skipping writing this sample. "
                f"Error: {e}"
            )

    writer.finish()


if __name__ == "__main__":
    main()
