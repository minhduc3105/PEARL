import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_bytes(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.partial-', suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def save_json(path, data):
    atomic_bytes(path, lambda tmp: Path(tmp).write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8'))

def save_png(path, array):
    atomic_bytes(path, lambda tmp: Image.fromarray(array.astype(np.uint8), mode='L').save(tmp))

def save_conf(path, array):
    atomic_bytes(path, lambda tmp: np.savez_compressed(tmp, confidence=array.astype(np.float32)))
