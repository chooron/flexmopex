"""Three-seed process evidence classification."""
import numpy as np

SEEDS = (42, 43, 44)


def classify(df):
    values = df.pivot(index=['gauge_id', 'process'], columns='seed', values='delta_J').reindex(columns=SEEDS)
    assert len(values) == 671 * 4 and np.isfinite(values.to_numpy()).all()
    a = values.to_numpy()
    median = np.median(a, axis=1)
    sd = np.std(a, axis=1, ddof=0)
    classes = np.where((a > 0).all(1) & (abs(median) > sd), 'decisive_open',
                       np.where((a < 0).all(1) & (abs(median) > sd), 'decisive_closed', 'uncertain'))
    result = values.rename(columns={s: f'delta_seed{s}' for s in SEEDS}).reset_index()
    result['delta_median'] = median
    result['delta_sd_ddof0'] = sd
    result['evidence_class'] = classes
    return result
