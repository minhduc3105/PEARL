"""Generate and evaluate PEARL pseudo masks on R2S100K."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .data import pairs, colorize
from .io import sha256, save_json, save_png, save_conf, atomic_bytes
from .metrics import evaluate, THRESHOLDS
from .taxonomy import CLASSES, NUM_CLASSES, model_vocabulary, model_to_dataset_ids

DEFAULT_IMAGES = '/run/media/minhduc/D/Study/r2s100k/train'
DEFAULT_LABELS = '/run/media/minhduc/D/Study/r2s100k/train_labels/Train-Labels'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / 'runs' / 'r2s100k_train'
CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'cfg_r2s100k.py'

def effective_config():
    from mmengine.config import Config
    cfg=Config.fromfile(str(CONFIG_PATH))
    return dict(model=dict(cfg.model), test_pipeline=[dict(step) for step in cfg.test_pipeline])


def file_stamp(path):
    p=Path(path); s=p.stat()
    return dict(path=str(p.resolve()), size=s.st_size, mtime_ns=s.st_mtime_ns)

def record_for(image, label):
    return dict(image=str(image.resolve()), label=str(label.resolve()) if label else None, image_stamp=file_stamp(image), label_stamp=file_stamp(label) if label else None, mask=f'masks/{image.name}.png', confidence=f'confidence/{image.name}.npz')

def vocabulary_text():
    prompts = model_vocabulary()
    ids = model_to_dataset_ids()
    return '\n'.join(', '.join(prompts[k] for k, v in enumerate(ids) if v == i) for i in range(NUM_CLASSES)) + '\n'

def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    sp=ap.add_subparsers(dest='command',required=True)
    for name in ('inspect','run','evaluate'):
        p=sp.add_parser(name)
        p.add_argument('--images-dir',default=DEFAULT_IMAGES)
        p.add_argument('--labels-dir',default=DEFAULT_LABELS,help='Use --no-labels for inference only')
        p.add_argument('--no-labels',action='store_true')
        p.add_argument('--max-images',type=int)
        if name!='inspect':
            p.add_argument('--output',default=str(DEFAULT_OUTPUT))
            p.add_argument('--filtered-threshold',type=float,choices=THRESHOLDS)
        if name=='run':
            p.add_argument('--checkpoint',default='ViT-B/16',help='Local CLIP checkpoint or built-in CLIP model name')
            p.add_argument('--resume',action='store_true')
            p.add_argument('--device',default='cuda')
    return ap

def preview(image_path, label_path, pred, output):
    from .data import load_gt
    with Image.open(image_path) as im:
        rgb=np.asarray(im.convert('RGB'))
    colored=colorize(pred)
    overlay=((rgb.astype(np.uint16)+colored.astype(np.uint16))//2).astype(np.uint8)
    panels=[rgb, colorize(load_gt(label_path)) if label_path else rgb, colored, overlay]
    atomic_bytes(output, lambda p: Image.fromarray(np.concatenate(panels,axis=1)).save(p))

def valid_output(root, rec, digests=None):
    try:
        if digests is not None:
            expected = digests.get(rec['image'])
            if not expected or expected.get('mask') != sha256(root/rec['mask']) or expected.get('confidence') != sha256(root/rec['confidence']):
                return False
        with Image.open(root/rec['mask']) as im:
            pred=np.asarray(im)
        with np.load(root/rec['confidence']) as z:
            conf=z['confidence']
        with Image.open(rec['image']) as im:
            size=im.size
        return pred.dtype==np.uint8 and pred.shape==(size[1],size[0]) and conf.shape==pred.shape and np.all(pred<NUM_CLASSES) and np.all(np.isfinite(conf)) and np.all((conf>=0)&(conf<=1))
    except (OSError, ValueError, KeyError, EOFError):
        return False

def make_manifest(args, items):
    labels=None if args.no_labels else args.labels_dir
    recs=[record_for(i,l) for i,l in items]
    return dict(schema=1, config=effective_config(), checkpoint=args.checkpoint, checkpoint_sha256=sha256(args.checkpoint) if Path(args.checkpoint).is_file() else sha256(Path.home()/'.cache/clip/ViT-B-16.pt') if args.checkpoint=='ViT-B/16' and (Path.home()/'.cache/clip/ViT-B-16.pt').is_file() else '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f' if args.checkpoint=='ViT-B/16' else None, vocabulary_sha256=hashlib.sha256(vocabulary_text().encode()).hexdigest(), taxonomy_sha256=sha256(Path(__file__).with_name('taxonomy.py')), config_sha256=sha256(CONFIG_PATH), images_dir=str(Path(args.images_dir).resolve()), labels_dir=str(Path(labels).resolve()) if labels else None, records=recs)

def load_model(checkpoint, vocab_path, device):
    import torch
    import mmcv, mmengine, mmseg
    from mmseg.utils import register_all_modules
    from mmseg.registry import MODELS
    from mmengine.config import Config
    import pearl_ovss  # register upstream model
    register_all_modules(init_default_scope=True)
    cfg=Config.fromfile(str(CONFIG_PATH))
    model_cfg=dict(cfg.model)
    model_cfg.update(clip_path=checkpoint,name_path=str(vocab_path),device=torch.device(device))
    model=MODELS.build(model_cfg)
    model.to(device)
    model.eval()
    return model, dict(torch=torch.__version__,mmcv=mmcv.__version__,mmengine=mmengine.__version__,mmseg=mmseg.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def main():
    args=parser().parse_args()
    if args.max_images is not None and args.max_images<=0:
        raise ValueError('--max-images must be positive')
    labels=None if args.no_labels else args.labels_dir
    if args.command=='inspect':
        items=pairs(args.images_dir,labels,args.max_images,validate=True)
        print(json.dumps(dict(images=len(items),with_gt=sum(l is not None for _,l in items),taxonomy_sha256=sha256(Path(__file__).with_name('taxonomy.py')),vocabulary=vocabulary_text().splitlines()),indent=2))
        return
    root=Path(args.output).expanduser()
    root=(PROJECT_ROOT/root).resolve() if not root.is_absolute() else root.resolve()
    if root != PROJECT_ROOT and PROJECT_ROOT not in root.parents:
        raise ValueError(f'Output must be inside the project: {PROJECT_ROOT}')
    if args.command=='evaluate':
        manifest=json.loads((root/'manifest.json').read_text())
        current=pairs(args.images_dir,labels,args.max_images,validate=False)
        expected=[record_for(i,l) for i,l in current]
        if expected!=manifest['records'][:len(expected)]:
            raise ValueError('Dataset files differ from manifest')
        records=manifest['records'][:len(expected)]
        progress=json.loads((root/'progress.json').read_text())
        if not all(valid_output(root,r,progress) for r in records):
            raise ValueError('Missing or invalid prediction output')
        result=evaluate(records,root,args.filtered_threshold) if labels else {'evaluated_images':0}
        print(json.dumps(result,indent=2))
        return
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    items=pairs(args.images_dir,labels,args.max_images,validate=True)
    manifest=make_manifest(args,items)
    manifest_path=root/'manifest.json'
    if root.exists() and any(root.iterdir()):
        if not args.resume:
            raise FileExistsError(f'{root} is not empty; use --resume or another output directory')
        if not manifest_path.is_file() or json.loads(manifest_path.read_text())!=manifest:
            raise ValueError('Resume manifest differs from dataset/checkpoint/config/vocabulary')
    else:
        root.mkdir(parents=True,exist_ok=True)
        save_json(manifest_path,manifest)
    vocab_path=root/'vocabulary.txt'
    if not vocab_path.exists():
        vocab_path.write_text(vocabulary_text(),encoding='utf-8')
    elif vocab_path.read_text(encoding='utf-8')!=vocabulary_text():
        raise ValueError('Vocabulary differs')
    progress_path=root/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    pending=[r for r in manifest['records'] if not valid_output(root,r,progress)]
    setup_start=time.perf_counter()
    model,versions=load_model(args.checkpoint,vocab_path,args.device) if pending else (None,{})
    load_s=time.perf_counter()-setup_start if pending else 0
    from mmengine.dataset import Compose
    if pending:
        from mmengine.config import Config
        pipeline=Compose(Config.fromfile(str(CONFIG_PATH)).test_pipeline)
    else:
        pipeline=None
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    infer_times=[]
    start=time.perf_counter()
    for n,r in enumerate(manifest['records']):
        if valid_output(root,r,progress):
            preview_path=root/'previews'/f"{Path(r['image']).name}.png"
            if n<20 and not preview_path.exists():
                with Image.open(root/r['mask']) as im:
                    preview(r['image'],r['label'],np.asarray(im).copy(),preview_path)
            continue
        data=pipeline(dict(img_path=r['image']))
        torch.cuda.synchronize() if args.device.startswith('cuda') else None
        t0=time.perf_counter()
        with torch.inference_mode():
            sample=model.test_step(dict(inputs=[data['inputs']],data_samples=[data['data_samples']]))[0]
        torch.cuda.synchronize() if args.device.startswith('cuda') else None
        infer_times.append(time.perf_counter()-t0)
        pred=sample.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
        probs=sample.seg_logits.data.float()
        confidence=probs.gather(0,torch.from_numpy(pred.astype(np.int64)).to(probs.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        save_png(root/r['mask'],pred)
        save_conf(root/r['confidence'],confidence)
        progress[r['image']]=dict(mask=sha256(root/r['mask']),confidence=sha256(root/r['confidence']))
        save_json(progress_path,progress)
        if n<20:
            preview(r['image'],r['label'],pred,root/'previews'/f'{Path(r["image"]).name}.png')
        if (n+1)%25==0 or n+1==len(items):
            print(f'{n+1}/{len(items)}: {Path(r["image"]).name}',flush=True)
    elapsed=time.perf_counter()-start
    if labels:
        result=evaluate(manifest['records'],root,args.filtered_threshold)
    else:
        result={'evaluated_images':0}
        if args.filtered_threshold is not None:
            for r in manifest['records']:
                with Image.open(root/r['mask']) as im:
                    selected=np.asarray(im).copy()
                with np.load(root/r['confidence']) as z:
                    selected[z['confidence']<args.filtered_threshold]=255
                save_png(root/'filtered'/f"{Path(r['image']).name}.png",selected)
    git=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False)
    output_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    run_path=root/'run.json'
    previous=json.loads(run_path.read_text()) if run_path.exists() else {}
    if infer_times or not previous:
        count=previous.get('inferences_total',0)+len(infer_times)
        infer_total=previous.get('inference_seconds_total',0)+sum(infer_times)
        wall_total=previous.get('wall_seconds',0)+elapsed
        save_json(run_path,dict(pearl_commit=git.stdout.strip() if git.returncode==0 else None,python=sys.version,platform=platform.platform(),versions=versions or previous.get('versions',{}),config=manifest['config'],checkpoint_sha256=manifest['checkpoint_sha256'],taxonomy_sha256=manifest['taxonomy_sha256'],config_sha256=manifest['config_sha256'],vocabulary_sha256=manifest['vocabulary_sha256'],images=len(items),inferences_total=count,model_load_seconds=previous.get('model_load_seconds',0)+load_s,inference_seconds_total=infer_total,inference_seconds_mean_excluding_first=float(np.mean(infer_times[1:])) if len(infer_times)>1 else previous.get('inference_seconds_mean_excluding_first'),wall_seconds=wall_total,throughput_images_per_second=count/wall_total if wall_total else None,peak_vram_bytes=max(previous.get('peak_vram_bytes') or 0,torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0),output_bytes=output_bytes))
    print(json.dumps({k:result.get(k) for k in ('miou_15','miou_road_14','evaluated_images')},indent=2))

if __name__=='__main__':
    main()
