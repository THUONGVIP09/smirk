"""
Compare two benchmark runs (occlusion_benchmark.py outputs) on the same test faces with a paired bootstrap.

For every dataset and protocol: score of A, score of B, difference B - A with its 95% interval, and whether the
interval excludes 0 (significant at the 5% level). Lower is better for expr_mm / expr_mm_visible / lmk_px, higher for
the emotion scores; exp_scale is better close to 1 (compared as |exp_scale - 1|).

    python -m src.evaluation.compare_runs --a results/original_transfer --b results/occlusion_aware_transfer \
        --out results/compare_transfer.csv
"""
import argparse
import numpy as np
import pandas as pd

from src.evaluation import bootstrap

METRICS = ['expr_mm', 'expr_mm_visible', 'lmk_px', 'exp_scale_error']
HIGHER_IS_BETTER = {'emotion_acc': True, 'emotion_mean_class_acc': True, 'expr_mm': False, 'expr_mm_visible': False,
                    'lmk_px': False, 'exp_scale_error': False}


def main():
    parser = argparse.ArgumentParser(description='Paired bootstrap comparison of two benchmark runs.')
    parser.add_argument('--a', required=True, help='Output folder of the reference run (per_image.csv)')
    parser.add_argument('--b', required=True, help='Output folder of the compared run')
    parser.add_argument('--out', default=None, help='CSV with the comparison')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()

    tables = []
    for path in [args.a, args.b]:
        table = pd.read_csv(f'{path}/per_image.csv', dtype={'name': str})
        table['exp_scale_error'] = (table['exp_scale'] - 1).abs()
        tables.append(table)
    keys = ['dataset', 'name', 'protocol']
    a, b = tables
    merged = a.merge(b, on=keys, suffixes=('_a', '_b'))
    assert len(merged) == len(a) == len(b), \
        f'the runs were not evaluated on the same faces ({len(a)} / {len(b)} rows, {len(merged)} in common): same --limit, inputs and protocols needed'

    groups = [(('all', p), g) for p, g in merged.groupby('protocol', sort=False)] + \
             list(merged.groupby(['dataset', 'protocol'], sort=False))
    real = merged[merged['protocol'].str.startswith('real_')]
    if len(real):
        groups += [(('all', 'real_all'), real)] + [((d, 'real_all'), g) for d, g in real.groupby('dataset', sort=False)]
    rng = np.random.default_rng(0)
    rows = []
    for (dataset, protocol), group in groups:
        index = bootstrap.resamples(len(group), args.bootstrap, rng)
        sampled = {}
        for side in ['a', 'b']:
            view = group[['label_a', f'pred_label_{side}'] + [f'{m}_{side}' for m in METRICS]]
            view.columns = ['label', 'pred_label'] + METRICS
            sampled[side] = (view, bootstrap.scores(view, index, METRICS))
        for m in bootstrap.SCORES + METRICS:
            if m not in sampled['a'][1]:
                continue  # e.g. expr_mm of the clean protocol (NaN)
            point = {}
            for side, (view, _) in sampled.items():
                if m in bootstrap.SCORES:
                    point[side] = bootstrap.accuracy(view['label'].to_numpy(), view['pred_label'].to_numpy(),
                                                     np.arange(len(view))[None])[m][0]
                else:
                    point[side] = float(view[m].mean())
            low, high = bootstrap.interval(sampled['b'][1][m] - sampled['a'][1][m])
            difference = point['b'] - point['a']
            significant = not (low <= 0 <= high)
            better = significant and (difference > 0) == HIGHER_IS_BETTER[m]
            rows.append({'dataset': dataset, 'protocol': protocol, 'metric': m, 'faces': len(group), 'a': point['a'],
                         'b': point['b'], 'b_minus_a': difference, 'ci_low': low, 'ci_high': high,
                         'significant': significant, 'verdict': ('B better' if better else 'B worse') if significant else 'no difference'})
    result = pd.DataFrame(rows)
    if args.out:
        result.to_csv(args.out, index=False)
    pd.set_option('display.width', 200)
    shown = result[(result['dataset'] == 'all') & result['metric'].isin(['emotion_mean_class_acc', 'expr_mm_visible'])]
    print(shown.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
