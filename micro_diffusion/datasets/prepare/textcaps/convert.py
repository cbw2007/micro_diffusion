import json
import os
import math
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from datasets import DownloadConfig, load_dataset
from PIL import Image
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
        "--raw_data_dir",
        type=str,
        default=None,
        help=(
            "Directory with manually downloaded TextCaps files. When set, convert "
            "uses local files and does not call HuggingFace load_dataset."
        ),
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
    parser.add_argument(
        "--download_timeout",
        type=int,
        default=3600,
        help="Total HTTP download timeout in seconds for HuggingFace datasets downloads.",
    )
    parser.add_argument(
        "--download_retries",
        type=int,
        default=5,
        help="Number of retries used by HuggingFace datasets downloads when supported.",
    )
    args = parser.parse_args()
    return args


def build_download_config(args: ArgumentParser) -> DownloadConfig:
    storage_options = {}
    try:
        import aiohttp

        storage_options = {
            "client_kwargs": {
                "timeout": aiohttp.ClientTimeout(total=args.download_timeout),
            }
        }
    except ImportError:
        print("aiohttp is unavailable; using datasets default HTTP timeout.")

    try:
        return DownloadConfig(
            max_retries=args.download_retries,
            storage_options=storage_options,
        )
    except TypeError:
        print("This datasets version does not support max_retries/storage_options; using default download config.")
        return DownloadConfig()


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
    if bbox_key is None and "ocr_info" in sample:
        bboxes = []
        for ocr_item in sample["ocr_info"]:
            bounding_box = ocr_item.get("bounding_box", {})
            if not isinstance(bounding_box, dict):
                continue
            raw_bbox = {
                "left": bounding_box.get("top_left_x", 0.0),
                "top": bounding_box.get("top_left_y", 0.0),
                "width": bounding_box.get("width", 0.0),
                "height": bounding_box.get("height", 0.0),
            }
            bbox = _normalize_bbox(
                raw_bbox,
                width=width,
                height=height,
                bbox_format=bbox_format,
            )
            if bbox is not None:
                bboxes.append(bbox)
        return bboxes
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


def _find_required_file(raw_data_dir: Path, filename: str) -> Path:
    matches = list(raw_data_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {raw_data_dir}. "
            "Download TextCaps raw JSON/image files before running convert."
        )
    return matches[0]


def _open_image_from_path_or_zip(
    image_name: str,
    images_dir: Optional[Path],
    images_zip: Optional[zipfile.ZipFile],
) -> Image.Image:
    image_filename = image_name if image_name.lower().endswith(".jpg") else f"{image_name}.jpg"
    if images_dir is not None:
        image_path = images_dir / image_filename
        if image_path.exists():
            with Image.open(image_path) as img:
                return img.convert("RGB").copy()
    if images_zip is not None:
        for member in (
            f"train_images/{image_filename}",
            f"test_images/{image_filename}",
            image_filename,
        ):
            try:
                with images_zip.open(member) as image_file:
                    with Image.open(image_file) as img:
                        return img.convert("RGB").copy()
            except KeyError:
                continue
    raise FileNotFoundError(f"Could not find image {image_filename}")


def _build_ocr_by_image_id(ocr_path: Path) -> Dict[str, Dict[str, Any]]:
    ocr_items = json.load(open(ocr_path, "r"))["data"]
    return {ocr_item["image_id"]: ocr_item for ocr_item in ocr_items}


def convert_from_raw_data_dir(args: ArgumentParser) -> None:
    raw_data_dir = Path(args.raw_data_dir)
    caption_paths = [
        _find_required_file(raw_data_dir, "TextCaps_0.1_train.json"),
        _find_required_file(raw_data_dir, "TextCaps_0.1_val.json"),
    ]
    ocr_paths = [
        _find_required_file(raw_data_dir, "TextVQA_Rosetta_OCR_v0.2_train.json"),
        _find_required_file(raw_data_dir, "TextVQA_Rosetta_OCR_v0.2_val.json"),
    ]

    images_dir_matches = list(raw_data_dir.rglob("train_images"))
    images_dir = images_dir_matches[0] if images_dir_matches else None
    images_zip_path = None
    zip_matches = list(raw_data_dir.rglob("train_val_images.zip"))
    if images_dir is None and zip_matches:
        images_zip_path = zip_matches[0]
    if images_dir is None and images_zip_path is None:
        raise FileNotFoundError(
            "Could not find train_images/ or train_val_images.zip under "
            f"{raw_data_dir}."
        )

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

    total_written = 0
    images_zip = zipfile.ZipFile(images_zip_path) if images_zip_path is not None else None
    try:
        for caption_path, ocr_path in zip(caption_paths, ocr_paths):
            captions = json.load(open(caption_path, "r"))["data"]
            ocr_by_image_id = _build_ocr_by_image_id(ocr_path)
            seen_image_ids = set()
            for caption_item in tqdm(captions, desc=f"Converting {caption_path.name}"):
                image_id = caption_item["image_id"]
                if image_id in seen_image_ids:
                    continue
                seen_image_ids.add(image_id)
                if not caption_item.get("reference_strs"):
                    continue

                image = _open_image_from_path_or_zip(
                    caption_item["image_name"],
                    images_dir=images_dir,
                    images_zip=images_zip,
                )
                mds_sample = {
                    "height": int(caption_item["image_height"]),
                    "width": int(caption_item["image_width"]),
                    "jpg": image,
                    "image_id": image_id,
                    "caption": caption_item["reference_strs"][0],
                }
                if args.save_text_bboxes:
                    sample = dict(caption_item)
                    sample["ocr_info"] = ocr_by_image_id.get(image_id, {}).get("ocr_info", [])
                    mds_sample["text_bboxes"] = json.dumps(
                        extract_text_bboxes(sample, bbox_format=args.bbox_format)
                    )
                writer.write(mds_sample)
                total_written += 1
    finally:
        if images_zip is not None:
            images_zip.close()
        writer.finish()
    print(f"Total {total_written} samples in textcaps dataset")


def main():
    args = parse_arguments()
    if args.raw_data_dir is not None:
        convert_from_raw_data_dir(args)
        return

    ds = load_dataset(
        "HuggingFaceM4/TextCaps",
        split="train+validation",
        download_config=build_download_config(args),
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
