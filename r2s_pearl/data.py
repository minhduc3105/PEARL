"""R2S100K pairing and exact mask decoding. GT is never used by inference."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
from PIL import Image
from .taxonomy import CLASSES, IMAGE_SUFFIXES, NUM_CLASSES

COLORS = np.asarray([CLASSES[i][2] for i in range(NUM_CLASSES)], dtype=np.uint8)

def pairs(images_dir, labels_dir=None, limit=None, validate=True):
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    images = sorted((p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name)
    if not images:
        raise ValueError(f'No images in {images_dir}')
    names = [p.name for p in images]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate image names')
    labels_dir = Path(labels_dir) if labels_dir else None
    if labels_dir and not labels_dir.is_dir():
        raise FileNotFoundError(labels_dir)
    if labels_dir:
        expected = {p.name + '___fuse.png' for p in images}
        actual = {p.name for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png'}
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise ValueError(f'GT mismatch: missing={len(missing)}, extra={len(extra)}; examples={list(sorted(missing))[:3]}, {list(sorted(extra))[:3]}')
    result = [(p, labels_dir / (p.name + '___fuse.png') if labels_dir else None) for p in images]
    result = result[:limit] if limit else result
    if validate:
        if len(result) > 100:
            with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
                list(pool.map(validate_pair, result, chunksize=8))
        else:
            for item in result:
                validate_pair(item)
    return result

def validate_pair(item):
    image, label = item
    with Image.open(image) as im:
        size = im.size
    if label:
        with Image.open(label) as im:
            if im.size != size:
                raise ValueError(f'Size mismatch: {image} and {label}')
            decode_mask(im, str(label))

def decode_mask(image, name='mask'):
    arr = np.asarray(image)
    if arr.ndim == 2 and arr.dtype == np.uint8:
        bad = np.unique(arr[arr >= NUM_CLASSES])
        if len(bad):
            raise ValueError(f'{name}: invalid class IDs {bad[:8].tolist()}')
        return arr.copy()
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f'{name}: expected RGB/RGBA or uint8 ID mask')
    if arr.shape[2] == 4 and not np.all(arr[:, :, 3] == 255):
        raise ValueError(f'{name}: transparent pixels are not valid GT')
    rgb = arr[:, :, :3]
    packed = rgb[:, :, 0].astype(np.uint32) << 16 | rgb[:, :, 1].astype(np.uint32) << 8 | rgb[:, :, 2].astype(np.uint32)
    palette = COLORS[:, 0].astype(np.uint32) << 16 | COLORS[:, 1].astype(np.uint32) << 8 | COLORS[:, 2].astype(np.uint32)
    order = np.argsort(palette)
    indexes = np.searchsorted(palette[order], packed)
    valid = indexes < NUM_CLASSES
    valid[valid] &= palette[order][indexes[valid]] == packed[valid]
    if not np.all(valid):
        bad = rgb[~valid][0].tolist()
        raise ValueError(f'{name}: invalid RGB color {bad}')
    return order[indexes].astype(np.uint8)

def load_gt(path):
    with Image.open(path) as im:
        return decode_mask(im, str(path))

def colorize(ids):
    return COLORS[ids]
