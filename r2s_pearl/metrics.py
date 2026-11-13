"""Dataset-level metrics from stored pseudo masks and confidence."""
import csv
from pathlib import Path
import numpy as np
from .taxonomy import CLASSES, NUM_CLASSES
from .data import load_gt
from .io import save_json

THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)

def confusion(gt, pred):
    if gt.shape != pred.shape:
        raise ValueError('Prediction and GT sizes differ')
    if np.any(pred >= NUM_CLASSES):
        raise ValueError('Prediction contains invalid ID')
    return np.bincount((gt.astype(np.int64) * NUM_CLASSES + pred).ravel(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)

def safe_ratio(a, b):
    return float(a / b) if b else None

def summary(cm):
    diag = np.diag(cm)
    gt = cm.sum(1)
    pred = cm.sum(0)
    union = gt + pred - diag
    rows = []
    for i in range(NUM_CLASSES):
        rows.append(dict(id=i, name=CLASSES[i][0], gt_pixels=int(gt[i]), predicted_pixels=int(pred[i]), intersection=int(diag[i]), union=int(union[i]), iou=safe_ratio(diag[i], union[i]), dice=safe_ratio(2*diag[i], gt[i]+pred[i]), precision=safe_ratio(diag[i], pred[i]), recall=safe_ratio(diag[i], gt[i])))
    vals = [r['iou'] for r in rows if r['iou'] is not None]
    roads = [r['iou'] for r in rows[1:] if r['iou'] is not None]
    recalls = [r['recall'] for r in rows if r['recall'] is not None]
    total = int(cm.sum())
    return dict(miou_15=float(np.mean(vals)) if vals else None, miou_road_14=float(np.mean(roads)) if roads else None, evaluated_classes_15=len(vals), evaluated_classes_road=len(roads), pixel_accuracy=safe_ratio(diag.sum(), total), mean_class_accuracy=float(np.mean(recalls)) if recalls else None, frequency_weighted_iou=float(sum((gt[i]/total)*rows[i]['iou'] for i in range(NUM_CLASSES) if rows[i]['iou'] is not None)) if total else None, total_pixels=total), rows

def write_csv(path, fields, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def evaluate(records, output, filtered_threshold=None):
    output = Path(output)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    threshold_stats = {str(t): dict(kept=0, correct=0, predicted=np.zeros(NUM_CLASSES, np.int64), tp=np.zeros(NUM_CLASSES, np.int64), gt_covered=np.zeros(NUM_CLASSES, np.int64)) for t in THRESHOLDS}
    bins = np.zeros((15, 3), dtype=np.float64)  # count, confidence sum, correct sum
    per_image = []
    gt_count = np.zeros(NUM_CLASSES, np.int64)
    for record in records:
        if not record['label']:
            continue
        gt = load_gt(record['label'])
        with Image.open(output / record['mask']) as im:
            pred = np.asarray(im).copy()
        with np.load(output / record['confidence']) as z:
            confidence = z['confidence']
        if confidence.shape != pred.shape or not np.all(np.isfinite(confidence)) or np.min(confidence) < 0 or np.max(confidence) > 1:
            raise ValueError(f"Invalid confidence: {record['confidence']}")
        one = confusion(gt, pred)
        cm += one
        gt_count += np.bincount(gt.ravel(), minlength=NUM_CLASSES)
        overall, _ = summary(one)
        per_image.append(dict(image=record['image'], miou_15=overall['miou_15'], miou_road_14=overall['miou_road_14'], pixel_accuracy=overall['pixel_accuracy']))
        correct = gt == pred
        index = np.minimum((confidence * 15).astype(np.intp), 14)
        bins[:, 0] += np.bincount(index.ravel(), minlength=15)
        bins[:, 1] += np.bincount(index.ravel(), weights=confidence.ravel(), minlength=15)
        bins[:, 2] += np.bincount(index.ravel(), weights=correct.ravel(), minlength=15)
        for t in THRESHOLDS:
            keep = confidence >= t
            s = threshold_stats[str(t)]
            s['kept'] += int(keep.sum())
            s['correct'] += int((keep & correct).sum())
            s['predicted'] += np.bincount(pred[keep], minlength=NUM_CLASSES)
            s['tp'] += np.bincount(pred[keep & correct], minlength=NUM_CLASSES)
            s['gt_covered'] += np.bincount(gt[keep & correct], minlength=NUM_CLASSES)
        if filtered_threshold is not None:
            selected = pred.copy()
            selected[confidence < filtered_threshold] = 255
            from .io import save_png
            save_png(output / 'filtered' / f"{Path(record['image']).name}.png", selected)
    result, rows = summary(cm)
    total = result['total_pixels']
    thresholds = []
    for t in THRESHOLDS:
        s = threshold_stats[str(t)]
        thresholds.append(dict(threshold=t, coverage=safe_ratio(s['kept'], total), retained_accuracy=safe_ratio(s['correct'], s['kept']), per_class=[dict(id=i, predicted_coverage=safe_ratio(s['predicted'][i], total), precision=safe_ratio(s['tp'][i], s['predicted'][i]), correct_gt_coverage=safe_ratio(s['gt_covered'][i], gt_count[i])) for i in range(NUM_CLASSES)]))
    nonempty = bins[:, 0] > 0
    ece = float(np.sum(np.abs(bins[nonempty, 2]/bins[nonempty, 0] - bins[nonempty, 1]/bins[nonempty, 0])*bins[nonempty, 0])/total) if total else None
    result.update(evaluated_images=len(per_image), thresholds=thresholds, ece_15_bins=ece)
    save_json(output / 'metrics.json', result)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([x['coverage'] for x in thresholds], [x['retained_accuracy'] for x in thresholds], 'o-')
        for x in thresholds:
            if x['coverage'] is not None and x['retained_accuracy'] is not None:
                ax.annotate(str(x['threshold']), (x['coverage'], x['retained_accuracy']))
        ax.set(xlabel='Pixel coverage', ylabel='Accuracy of retained pixels', xlim=(0, 1), ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(output / 'quality_coverage.png', dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    write_csv(output / 'per_class.csv', list(rows[0]), rows)
    write_csv(output / 'per_image.csv', ['image','miou_15','miou_road_14','pixel_accuracy'], per_image)
    write_csv(output / 'confusion.csv', ['gt_id']+[str(i) for i in range(NUM_CLASSES)], [dict(gt_id=i, **{str(j):int(cm[i,j]) for j in range(NUM_CLASSES)}) for i in range(NUM_CLASSES)])
    return result

from PIL import Image
