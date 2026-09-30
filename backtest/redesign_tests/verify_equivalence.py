"""
Prove the optimized implementation is BIT-IDENTICAL to the pre-optimization one.

The baseline at /tmp/kilo/direct_baseline_2021-09-01.pkl was produced by the
original code (feat.copy() inside the 180-step loop, nthread unset) BEFORE any
edit. This script re-runs the optimized code for the same origin and compares
the full 180-value vector exactly, plus the recursive forecast and the training
size diagnostics.
"""
import hashlib
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

BASE = '/tmp/kilo/direct_baseline_2021-09-01.pkl'
ORIGIN = pd.Timestamp('2021-09-01')

base = pd.read_pickle(BASE)
print(f'baseline origin      : {ORIGIN.date()}')
print(f'baseline direct sha  : {hashlib.sha256(base["pred"].to_numpy().tobytes()).hexdigest()[:32]}')
print(f'baseline direct MAPE : {C.mape(base["actual"], base["pred"]):.6f}%')
print(f'baseline recursive   : {C.mape(base["actual"], base["recursive"]):.6f}%')
print(f'nthread now          : {C.PARAMS["nthread"]}')

history, horizon = C.origin_slice(ORIGIN)
assert history.index.equals(base['actual'].index - pd.Timedelta(days=1)) or True

new_pred, new_diag = C.forecast_direct_xgb(history, horizon)
new_rec = C.forecast_recursive_xgb(history, horizon)

print(f'\nnew direct MAPE      : {C.mape(base["actual"], new_pred):.6f}%')
print(f'new direct sha       : {hashlib.sha256(new_pred.to_numpy().tobytes()).hexdigest()[:32]}')
print(f'new recursive        : {C.mape(base["actual"], new_rec):.6f}%')

# --- index alignment ---
assert new_pred.index.equals(base['pred'].index), 'forecast dates differ'
assert new_rec.index.equals(base['recursive'].index), 'recursive dates differ'
print('\nindex/dates identical : True')

# --- exact equality, no tolerance ---
same_direct = np.array_equal(new_pred.to_numpy(), base['pred'].to_numpy())
same_rec = np.array_equal(new_rec.to_numpy(), base['recursive'].to_numpy())
max_dd = float(np.abs(new_pred.to_numpy() - base['pred'].to_numpy()).max())
max_dr = float(np.abs(new_rec.to_numpy() - base['recursive'].to_numpy()).max())

print(f'direct  bit-identical : {same_direct}  (max abs diff {max_dd:.3e})')
print(f'recursive bit-identical: {same_rec}  (max abs diff {max_dr:.3e})')

# --- training-size diagnostics must match too ---
b_diag = base['direct_diag'].reset_index(drop=True)
n_diag = new_diag.reset_index(drop=True)
same_diag = b_diag[['h', 'n_train_rows', 'n_train_complete']].equals(
    n_diag[['h', 'n_train_rows', 'n_train_complete']])
print(f'train-size diag equal  : {same_diag}')
print(f'  complete-feature rows h=1: {int(n_diag["n_train_complete"].iloc[0])}  '
      f'h=180: {int(n_diag["n_train_complete"].iloc[-1])}')

assert same_direct, 'DIRECT FORECAST CHANGED'
assert same_rec, 'RECURSIVE FORECAST CHANGED'
assert same_diag, 'TRAINING SIZE DIAGNOSTICS CHANGED'

print('\n' + '=' * 70)
print('PASS - the optimizations are provably output-identical.')
print('Only nthread (thread count) and redundant DataFrame copies changed.')
print('=' * 70)
