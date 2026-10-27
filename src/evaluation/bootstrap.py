"""
Bootstrap confidence intervals of the benchmark scores (resampling the test faces).

Every score is computed on B resamples of the faces (with replacement); the 95% interval is the 2.5 / 97.5 percentile
of the resampled scores. Differences between two settings evaluated on the SAME faces (a protocol vs clean, model B vs
model A) use the same resamples for both (paired bootstrap), which gives much tighter intervals than two separate ones.
A difference whose interval does not contain 0 is significant at the 5% level.
"""
import numpy as np

SCORES = ['emotion_acc', 'emotion_mean_class_acc']


def resamples(n, count, rng):
    return rng.integers(0, n, size=(count, n))


def accuracy(labels, predictions, index):
    """Accuracy and mean class accuracy of every resample (index: B x n). Classes missing from a resample are skipped."""
    correct = (labels == predictions)[index]
    sampled = labels[index]
    per_class = []
    for c in np.unique(labels):
        members = sampled == c
        count = members.sum(1)
        per_class.append(np.where(count > 0, (correct & members).sum(1) / np.maximum(count, 1), np.nan))
    return {'emotion_acc': correct.mean(1), 'emotion_mean_class_acc': np.nanmean(np.stack(per_class), axis=0)}


def scores(table, index, metrics):
    """Resampled emotion scores and mean metrics of a per-image table (rows in the order of index)."""
    out = accuracy(table['label'].to_numpy(), table['pred_label'].to_numpy(), index)
    for m in metrics:
        values = table[m].to_numpy(dtype=np.float64)
        if np.isnan(values).all():
            continue
        with np.errstate(invalid='ignore'):
            out[m] = np.nanmean(values[index], axis=1)
    return out


def interval(values):
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
