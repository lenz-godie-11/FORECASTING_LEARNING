"""Figures for the walk-forward backtest."""
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = 'backtest'
per_origin = pd.read_pickle(f'{OUT}/per_origin.pkl')
bands = pd.read_csv(f'{OUT}/backtest_horizon_bands.csv')
summary = pd.read_csv(f'{OUT}/backtest_summary.csv')
results = pd.read_csv(f'{OUT}/backtest_results.csv')

LABEL = {'XGB': 'XGBoost (recursive)', 'SeasNaive': 'Seasonal naive (7d)',
         'Weekday8': 'Weekday mean (8w)', 'LastValue': 'Last value'}

# ---------------------------------------------------------------- Figure 1
# Actual vs predicted trajectory for four representative origins, chosen by
# backtest quality rather than by hand.
ranked = results.set_index('Origin')['XGB_MAPE'].sort_values()
picks = {'Best': ranked.index[0], 'Median': ranked.index[len(ranked) // 2],
         'Worst': ranked.index[-1], 'Most recent': results['Origin'].iloc[-1]}

fig, axes = plt.subplots(2, 2, figsize=(17, 9), sharey=True)
for ax, (tag, origin) in zip(axes.ravel(), picks.items()):
    d = per_origin[pd.Timestamp(origin)]
    a, p = d['actual'], d['xgb']
    mape = float(np.mean(np.abs((a - p) / a)) * 100)
    ax.plot(a.index, a, lw=1.4, color='0.25', label='actual')
    ax.plot(p.index, p, lw=1.8, color='crimson', label='XGBoost recursive')
    ax.plot(d['wm'].index, d['wm'], lw=1.1, ls='--', color='navy',
            label='weekday mean baseline')
    ax.axvline(a.index[0], color='k', ls=':', lw=1)
    ax.set_title(f'{tag}: origin {origin}   MAPE {mape:.1f}%', fontsize=11, weight='bold')
    ax.set_xlabel('date')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=.3)
axes[0, 0].set_ylabel('patients / day')
axes[1, 0].set_ylabel('patients / day')
fig.suptitle('Walk-forward backtest: actual vs 180-day recursive forecast\n'
             'dotted line = forecast origin (last day of training data)',
             fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_trajectories.png', dpi=100)
plt.close()
print('wrote fig1_trajectories.png  origins:', picks)

# ---------------------------------------------------------------- Figure 2
# Error as a function of horizon length, all methods.
order = ['1-30', '31-60', '61-90', '91-120', '121-150', '151-180']
x = np.arange(len(order))
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for m in ['XGB', 'SeasNaive', 'Weekday8', 'LastValue']:
    sub = bands[bands.Method == m].set_index('band').reindex(order)
    axes[0].plot(x, sub['MAPE'], marker='o', lw=2,
                 color='crimson' if m == 'XGB' else None,
                 label=LABEL[m])
    axes[1].plot(x, sub['RMSE'], marker='o', lw=2,
                 color='crimson' if m == 'XGB' else None,
                 label=LABEL[m])
for ax, ttl in zip(axes, ['MAPE by horizon band', 'RMSE by horizon band']):
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30)
    ax.set_xlabel('forecast horizon (days ahead)')
    ax.set_ylabel('% error' if 'MAPE' in ttl else 'patients')
    ax.set_title(ttl, weight='bold')
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
fig.suptitle('Forecast error grows with horizon length; XGBoost is the worst of the '
             'competent methods at every horizon', fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_error_by_horizon.png', dpi=100)
plt.close()
print('wrote fig2_error_by_horizon.png')

# ---------------------------------------------------------------- Figure 3
# Per-origin MAPE, XGBoost vs the best baseline.
comp = results[['Origin', 'XGB_MAPE', 'SeasNaive_MAPE', 'Weekday8_MAPE',
                'LastValue_MAPE']].set_index('Origin')
comp = comp.reindex(sorted(comp.index))
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
w = 0.27
xs = np.arange(len(comp))
for i, (c, col) in enumerate([('XGB_MAPE', 'crimson'), ('SeasNaive_MAPE', 'darkorange'),
                              ('Weekday8_MAPE', 'navy')]):
    axes[0].bar(xs + (i - 1) * w, comp[c], w, label=LABEL[
        {'XGB_MAPE': 'XGB', 'SeasNaive_MAPE': 'SeasNaive',
         'Weekday8_MAPE': 'Weekday8'}[c]], color=col)
    axes[1].bar(xs + (i - 1) * w, comp[c] - comp['XGB_MAPE'], w, color=col,
                label=LABEL[{'XGB_MAPE': 'XGB', 'SeasNaive_MAPE': 'SeasNaive',
                              'Weekday8_MAPE': 'Weekday8'}[c]])
axes[0].set_xticks(xs)
axes[0].set_xticklabels(comp.index, rotation=60, fontsize=8)
axes[0].set_ylabel('180-day MAPE (%)')
axes[0].set_title('Per-origin 180-day error', weight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=.3, axis='y')
axes[1].axhline(0, color='k', lw=1)
axes[1].set_xticks(xs)
axes[1].set_xticklabels(comp.index, rotation=60, fontsize=8)
axes[1].set_ylabel('MAPE minus XGBoost (pp)')
axes[1].set_title('Advantage over XGBoost (negative = baseline wins)', weight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=.3, axis='y')
fig.suptitle('XGBoost loses to both simple baselines at 13 of 14 origins', fontsize=12,
             weight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_per_origin.png', dpi=100)
plt.close()
print('wrote fig3_per_origin.png')

wins = int((comp['XGB_MAPE'] < comp['Weekday8_MAPE']).sum())
wins_sn = int((comp['XGB_MAPE'] < comp['SeasNaive_MAPE']).sum())
wins_lv = int((comp['XGB_MAPE'] < comp['LastValue_MAPE']).sum())
print(f'\nXGBoost beats weekday-mean at {wins}/{len(comp)} origins')
print(f'XGBoost beats seasonal-naive at {wins_sn}/{len(comp)} origins')
print(f'XGBoost beats last-value  at {wins_lv}/{len(comp)} origins')
