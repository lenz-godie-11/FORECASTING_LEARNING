"""Fine-grained horizon analysis + statistical comparison + figures."""
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
res = pd.read_csv(f'{HERE}/redesign_results.csv')
bands = pd.read_csv(f'{HERE}/redesign_bands.csv')
store = pd.read_pickle(f'{HERE}/redesign_per_origin.pkl')
dt = pd.read_csv(f'{HERE}/direct_train_sizes.csv')

LABEL = {
    'XGB_direct': 'XGBoost direct (multi-horizon)',
    'XGB_recursive': 'XGBoost recursive (current production)',
    'Theta_p7': 'Theta (period 7)',
    'ETS_damped_seas7': 'ETS damped + weekly seasonal',
    'ETS_trend_seas7': 'ETS trend + weekly seasonal',
    'ETS_seasonal_only': 'ETS weekly seasonal only',
    'Weekday8': 'Weekday mean (8w)',
    'SeasNaive': 'Seasonal naive (7d)',
    'LastValue': 'Last value',
    'Holt_damped': 'Holt damped',
    'Holt_plain': 'Holt plain',
    'Theta_p30': 'Theta (period 30)',
}
SHORT = ['XGB_direct', 'XGB_recursive', 'Theta_p7', 'ETS_damped_seas7', 'Weekday8',
         'SeasNaive', 'LastValue']

pd.set_option('display.width', 250, 'display.max_columns', 60)

# ---------------------------------------------------------------- 1. overall
print('=' * 104)
print('1. OVERALL 180-DAY PERFORMANCE (mean / median across 14 origins)')
print('=' * 104)
summ = pd.read_csv(f'{HERE}/redesign_summary.csv')
summ['label'] = summ.Method.map(LABEL)
print(summ.sort_values('MAPE')[['label', 'RMSE', 'MAE', 'MAPE', 'sMAPE', 'MAPE_med']]
      .rename(columns={'label': 'Method', 'MAPE_med': 'MAPE (median)'}).round(2)
      .to_string(index=False))

# ------------------------------------------------- 2. paired significance
print('\n' + '=' * 104)
print('2. PAIRED SIGNIFICANCE (Wilcoxon signed-rank over the 14 origins, MAPE)')
print('=' * 104)
pairs = [('XGB_direct', 'XGB_recursive'), ('XGB_direct', 'Theta_p7'),
         ('XGB_direct', 'Weekday8'), ('XGB_recursive', 'Theta_p7'),
         ('XGB_recursive', 'Weekday8'), ('Theta_p7', 'Weekday8'),
         ('Theta_p7', 'SeasNaive')]
sig = []
for a, b in pairs:
    x, y = res[f'{a}_MAPE'].to_numpy(), res[f'{b}_MAPE'].to_numpy()
    d = x - y
    _, p = stats.wilcoxon(x, y)
    wins = int((d < 0).sum())
    better = a if d.mean() < 0 else b
    sig.append(dict(Comparison=f'{LABEL[a]} vs {LABEL[b]}',
                    mean_diff_pp=round(d.mean(), 2), wins=f'{wins}/14',
                    better=LABEL[better], p_value=round(p, 4)))
    print(f'  {LABEL[a]:<36} vs {LABEL[b]:<34} diff {d.mean():+6.2f}pp  '
          f'wins {wins:>2}/14  p={p:.4f}  better: {LABEL[better]}')
pd.DataFrame(sig).to_csv(f'{HERE}/redesign_significance.csv', index=False)

# --------------------------------------------- 3. per-horizon (14-day bins)
print('\n' + '=' * 104)
print('3. FINE-GRAINED MAPE BY 14-DAY HORIZON BIN (mean across 14 origins)')
print('=' * 104)
EDGE = list(range(1, 181, 14))
rows = []
for origin, po in store['preds'].items():
    a = store['actual'][origin].to_numpy()
    for name, p in po.items():
        e = np.abs((a - p.to_numpy()) / a) * 100
        for lo in EDGE:
            hi = min(lo + 13, 180)
            sl = slice(lo - 1, hi)
            rows.append(dict(Origin=origin.date(), Method=name, lo=lo,
                             MAPE=e[sl].mean()))
fine = pd.DataFrame(rows)
piv = fine.pivot_table(index='Method', columns='lo', values='MAPE')
piv = piv.reindex([m for m in LABEL if m in piv.index])
print('bins: ' + '  '.join(f'{lo}-{min(lo+13,180)}' for lo in EDGE))
print(piv.round(2).to_string())
piv.to_csv(f'{HERE}/redesign_fine_horizon.csv')

# ------------------------------------------------------ 4. practical horizon
print('\n' + '=' * 104)
print('4. WHERE DOES THE FORECAST BECOME UNUSABLE? (first bin exceeding each MAPE threshold)')
print('=' * 104)
rows = []
for thr in [15, 18, 20, 22, 25]:
    r = {'MAPE threshold': f'>{thr}%'}
    for m in SHORT:
        s = piv.loc[m].dropna()
        bad = [lo for lo in s.index if s[lo] > thr]
        r[m] = f'{bad[0]}' if bad else 'never'
    rows.append(r)
ph = pd.DataFrame(rows).set_index('MAPE threshold')
print(ph.to_string())
ph.to_csv(f'{HERE}/redesign_practical_horizon.csv')

# error growth factor
print('\nerror growth, days 1-14 -> days 167-180:')
for m in SHORT:
    s = piv.loc[m].dropna()
    print(f'  {LABEL[m]:<36} {s.iloc[0]:6.2f}% -> {s.iloc[-1]:6.2f}%  '
          f'({s.iloc[-1]/s.iloc[0]:.2f}x)')
pd.Series({m: piv.loc[m, EDGE[-1]] / piv.loc[m, EDGE[0]] for m in SHORT}).rename(
    'growth').to_csv(f'{HERE}/redesign_growth.csv')

# ------------------------- 5. direct model: long-horizon training size effect
print('\n' + '=' * 104)
print('5. DIRECT MODEL: is its long-horizon weakness a DATA-SIZE problem?')
print('=' * 104)
early = ['2021-06-01', '2021-08-01', '2021-10-01', '2021-12-01']
late = ['2022-02-01', '2022-04-01', '2022-06-01', '2022-07-01']
d2 = fine[(fine.Method == 'XGB_direct')]
for tag, sub in [('early origins (380-563d train)', early), ('late origins (625-775d train)', late)]:
    m = d2[d2.Origin.astype(str).isin(sub)]
    last = m[m.lo >= 121]['MAPE'].mean()
    first = m[m.lo <= 28]['MAPE'].mean()
    print(f'  {tag:<34} direct MAPE 1-28d {first:6.2f}%  121-180d {last:6.2f}%  '
          f'growth {last/first:.2f}x')
g = dt.groupby('Origin').agg(min_complete=('n_train_complete', 'min'),
                             max_complete=('n_train_complete', 'max'))
print(f'\n  complete-feature training rows for the h=180 model: '
      f'{int(g["min_complete"].min())} .. {int(g["min_complete"].max())} across origins')

# ---------------------------------------------------------------- 6. figures
# Fig 1: overall MAPE bars
fig, ax = plt.subplots(figsize=(12, 6))
s = summ.sort_values('MAPE')
cols = ['crimson' if 'recursive' in m else 'darkorange' if 'direct' in m
        else 'seagreen' if m.startswith('Theta') or m.startswith('ETS')
        else 'navy' if m == 'Weekday8' else 'grey' for m in s.Method]
ax.barh([LABEL[m] for m in s.Method], s.MAPE, color=cols)
ax.axvline(s.MAPE.iloc[0], color='k', ls=':', lw=1)
ax.set_xlabel('MAPE (%) over the full 180-day horizon, mean of 14 origins')
ax.set_title('All methods, 180-day walk-forward backtest', weight='bold')
ax.grid(alpha=.3, axis='x')
for i, v in enumerate(s.MAPE):
    ax.text(v + .3, i, f'{v:.2f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{HERE}/fig_overall_mape.png', dpi=100)
plt.close()

# Fig 2: MAPE vs horizon, 14-day bins
fig, ax = plt.subplots(figsize=(13, 6))
for m in ['XGB_recursive', 'XGB_direct', 'Theta_p7', 'ETS_damped_seas7', 'Weekday8']:
    s = piv.loc[m]
    ax.plot(s.index, s.values, marker='o', lw=2,
            ls='--' if m == 'XGB_recursive' else '-', label=LABEL[m])
for thr, c in [(20, 'darkorange'), (25, 'firebrick')]:
    ax.axhline(thr, color=c, ls=':', lw=1)
    ax.text(2, thr + .3, f'{thr}% MAPE', color=c, fontsize=8)
ax.set_xlabel('forecast horizon (days ahead, 14-day bins)')
ax.set_ylabel('MAPE (%)')
ax.set_xticks(EDGE[::2])
ax.set_xticklabels([f'{lo}' for lo in EDGE[::2]])
ax.set_title('Error growth with horizon: recursive XGBoost diverges, direct and Theta hold up',
             weight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f'{HERE}/fig_error_vs_horizon.png', dpi=100)
plt.close()

# Fig 3: per-origin MAPE, best methods
comp = res[['Origin'] + [f'{m}_MAPE' for m in
                         ['XGB_direct', 'XGB_recursive', 'Theta_p7', 'ETS_damped_seas7',
                          'Weekday8']]].set_index('Origin').sort_index()
fig, ax = plt.subplots(figsize=(13, 6))
comp.plot(kind='bar', ax=ax, width=.82)
ax.set_ylabel('180-day MAPE (%)')
ax.set_title('Per-origin 180-day MAPE', weight='bold')
ax.legend([LABEL[m] for m in
           ['XGB_direct', 'XGB_recursive', 'Theta_p7', 'ETS_damped_seas7', 'Weekday8']],
          fontsize=8)
ax.grid(alpha=.3, axis='y')
plt.tight_layout()
plt.savefig(f'{HERE}/fig_per_origin.png', dpi=100)
plt.close()

# Fig 4: representative trajectory, best & worst origin for the top method
r = res.set_index('Origin')
best_o = r.Theta_p7_MAPE.idxmin()
worst_o = r.Theta_p7_MAPE.idxmax()
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
for ax, o, tag in zip(axes, [best_o, worst_o], ['best origin', 'worst origin']):
    d = store['preds'][pd.Timestamp(o)]
    a = store['actual'][pd.Timestamp(o)]
    ax.plot(a.index, a, lw=1.4, color='0.25', label='actual')
    ax.plot(d['Theta_p7'].index, d['Theta_p7'], lw=1.8, color='seagreen',
            label='Theta (period 7)')
    ax.plot(d['XGB_direct'].index, d['XGB_direct'], lw=1.4, color='darkorange',
            label='XGBoost direct')
    ax.plot(d['XGB_recursive'].index, d['XGB_recursive'], lw=1.2, ls='--',
            color='crimson', label='XGBoost recursive')
    ax.axvline(a.index[0], color='k', ls=':')
    ax.set_title(f'{tag}: origin {o}', weight='bold')
    ax.set_xlabel('date')
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
axes[0].set_ylabel('patients / day')
fig.suptitle('180-day forecasts at the Theta optimum and the Theta worst case', weight='bold')
plt.tight_layout()
plt.savefig(f'{HERE}/fig_trajectories.png', dpi=100)
plt.close()

print(f"\nbest Theta origin {best_o} ({r.Theta_p7_MAPE.min():.2f}%), "
      f"worst {worst_o} ({r.Theta_p7_MAPE.max():.2f}%)")
print('figures written: fig_overall_mape, fig_error_vs_horizon, fig_per_origin, '
      'fig_trajectories')
