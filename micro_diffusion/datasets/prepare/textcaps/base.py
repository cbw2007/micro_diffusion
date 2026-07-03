import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from streaming import Stream, StreamingDataset

from micro_diffusion.models.utils import UniversalTokenizer


def _resize_center_crop_bboxes(
    bboxes: List[List[float]],
    orig_width: int,
    orig_height: int,
    target_size: int,
) -> List[List[float]]:
    scale = target_size / min(orig_width, orig_height)
    resized_width = orig_width * scale
    resized_height = orig_height * scale
    crop_left = max(0.0, (resized_width - target_size) / 2)
    crop_top = max(0.0, (resized_height - target_size) / 2)

    transformed = []
    for x0, y0, x1, y1 in bboxes:
        tx0 = x0 * scale - crop_left
        tx1 = x1 * scale - crop_left
        ty0 = y0 * scale - crop_top
        ty1 = y1 * scale - crop_top
        tx0 = max(0.0, min(float(target_size), tx0))
        tx1 = max(0.0, min(float(target_size), tx1))
        ty0 = max(0.0, min(float(target_size), ty0))
        ty1 = max(0.0, min(float(target_size), ty1))
        if tx1 > tx0 and ty1 > ty0:
            transformed.append([tx0, ty0, tx1, ty1])
    return transformed


def _build_text_region_mask(
    bboxes: List[List[float]],
    orig_width: int,
    orig_height: int,
    target_size: int,
    text_region_dilate_px: int,
) -> torch.Tensor:
    latent_size = target_size // 8
    mask = np.zeros((latent_size, latent_size), dtype=np.uint8)
    if len(bboxes) == 0:
        return torch.from_numpy(mask)

    transformed = _resize_center_crop_bboxes(
        bboxes=bboxes,
        orig_width=orig_width,
        orig_height=orig_height,
        target_size=target_size,
    )
    scale = latent_size / target_size
    for x0, y0, x1, y1 in transformed:
        x0 = max(0.0, x0 - text_region_dilate_px)
        y0 = max(0.0, y0 - text_region_dilate_px)
        x1 = min(float(target_size), x1 + text_region_dilate_px)
        y1 = min(float(target_size), y1 + text_region_dilate_px)
        lx0 = max(0, min(latent_size, int(math.floor(x0 * scale))))
        ly0 = max(0, min(latent_size, int(math.floor(y0 * scale))))
        lx1 = max(0, min(latent_size, int(math.ceil(x1 * scale))))
        ly1 = max(0, min(latent_size, int(math.ceil(y1 * scale))))
        if lx1 > lx0 and ly1 > ly0:
            mask[ly0:ly1, lx0:lx1] = 1
    return torch.from_numpy(mask)


class StreamingTextcapsDatasetForPreCompute(StreamingDataset):
    """Streaming dataset that resizes images to user-provided resolutions and tokenizes captions."""

    def __init__(
        self,
        streams: Sequence[Stream],
        transforms_list: List[Callable],
        batch_size: int,
        tokenizer_name: str,
        shuffle: bool = False,
        caption_key: str = 'caption_syn_pixart_llava15',
        save_text_region_masks: bool = False,
        text_region_dilate_px: int = 4,
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        self.transforms_list = transforms_list
        self.caption_key = caption_key
        self.save_text_region_masks = save_text_region_masks
        self.text_region_dilate_px = text_region_dilate_px
        self.tokenizer = UniversalTokenizer(tokenizer_name)
        print("Created tokenizer:", tokenizer_name)
        assert self.transforms_list is not None, 'Must provide transforms to resize and center crop images'

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        ret = {}

        out = self.tokenizer.tokenize(sample[self.caption_key])
        ret[self.caption_key] = out['input_ids'].clone().detach()
        if 'attention_mask' in out:
            ret[f'{self.caption_key}_attention_mask'] = out['attention_mask'].clone().detach()

        for i, transform in enumerate(self.transforms_list):
            img = sample['jpg']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = transform(img)
            ret[f'image_{i}'] = img
            if self.save_text_region_masks:
                text_bboxes = json.loads(sample.get('text_bboxes', '[]'))
                target_size = img.shape[-1]
                ret[f'text_region_mask_{i}'] = _build_text_region_mask(
                    bboxes=text_bboxes,
                    orig_width=int(sample['width']),
                    orig_height=int(sample['height']),
                    target_size=target_size,
                    text_region_dilate_px=self.text_region_dilate_px,
                )

        ret['sample'] = sample
        return ret


def build_streaming_textcaps_precompute_dataloader(
    datadir: Union[List[str], str],
    batch_size: int,
    resize_sizes: Optional[List[int]] = None,
    drop_last: bool = False,
    shuffle: bool = True,
    caption_key: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
    save_text_region_masks: bool = False,
    text_region_dilate_px: int = 4,
    **dataloader_kwargs,
) -> DataLoader:
    """Builds a streaming mds dataloader returning multiple image sizes and text captions."""
    assert resize_sizes is not None, 'Must provide target resolution for image resizing'
    
    datadir = [datadir] if isinstance(datadir, str) else datadir
    streams = [Stream(remote=None, local=path) for path in datadir]

    transforms_list = []
    for size in resize_sizes:
        transforms_list.append(
            transforms.Compose([
                transforms.Resize(
                    size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )

    dataset = StreamingTextcapsDatasetForPreCompute(
        streams=streams,
        shuffle=shuffle,
        transforms_list=transforms_list,
        batch_size=batch_size,
        caption_key=caption_key,
        tokenizer_name=tokenizer_name,
        save_text_region_masks=save_text_region_masks,
        text_region_dilate_px=text_region_dilate_px,
    )

    def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        out = {k: [] for k in batch[0].keys()}
        for sample in batch:
            for k, v in sample.items():
                out[k].append(v)
        return out

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=custom_collate,
        **dataloader_kwargs,
    )

    return dataloader
