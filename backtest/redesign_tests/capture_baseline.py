"""Capture the CURRENT (pre-optimization) direct multi-horizon forecast for the
smoke-test origin, so the optimized version can be proven bit-identical.

Run this BEFORE editing common.py.
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ORIGIN = pd.Timestamp('2021-09-01')
OUT = '/tmp/kilo/direct_baseline_2021-09-01.pkl'

history, horizon = C.origin_slice(ORIGIN)
print(f'origin      : {ORIGIN.date()}')
print(f'history     : {len(history)} days, {history.index.min().date()} -> '
      f'{history.index.max().date()}')
print(f'horizon     : {len(horizon)} days, {horizon.min().date()} -> {horizon.max().date()}')
print(f'PARAMS keys : {sorted(C.PARAMS)}')
print('nthread in PARAMS:', C.PARAMS.get('nthread', '(not set -> library default)'))

pred, diag = C.forecast_direct_xgb(history, horizon)

rec = C.forecast_recursive_xgb(history, horizon)

payload = dict(origin=ORIGIN, pred=pred, direct_diag=diag, recursive=rec,
               actual=C.DF.loc[horizon, C.TARGET])
pd.to_pickle(payload, OUT)

np.save('/tmp/kilo/direct_baseline_values.npy', pred.to_numpy())
np.save('/tmp/kilo/recursive_baseline_values.npy', rec.to_numpy())

print(f'\ndirect     MAPE {C.mape(payload["actual"], pred):.6f}%')
print(f'recursive  MAPE {C.mape(payload["actual"], rec):.6f}%')
print(f'\nsaved -> {OUT}')
print('first 5 direct:', pred.head().to_numpy())
print('last 5 direct :', pred.tail().to_numpy())
print('sha of direct values:',
      __import__('hashlib').sha256(pred.to_numpy().tobytes()).hexdigest()[:32])
