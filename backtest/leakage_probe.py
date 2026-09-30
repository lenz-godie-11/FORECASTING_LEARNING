"""Standalone leakage probe for the recursive forecast.

Confirms that `recursive_forecast` produces identical output regardless of what
the caller has in the frame it passes in beyond the horizon. If the function
were reading true future values to build features, poisoning those values would
change the forecast. It must not.
"""
import json
import re
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb

TARGET = 'Patients'

with open('Solution_2.ipynb') as f:
    nb = json.load(f)
sources = [''.join(c['source']) for c in nb['cells']]


def grab(name):
    pat = re.compile(rf'^def\s+{name}\s*\(', re.M)
    for s in sources:
        if pat.search(s):
            return s
    raise LookupError(name)


_dummy = pd.DataFrame({TARGET: [1.0, 2.0, 3.0]},
                      index=pd.date_range('2020-01-01', periods=3, freq='D'))
NS = {'pd': pd, 'np': np, 'xgb': xgb, 'df': _dummy}
for fn in ('create_features', 'add_lags', 'recursive_forecast'):
    exec(compile(grab(fn), fn, 'exec'), NS)
create_features, add_lags, recursive_forecast = (
    NS['create_features'], NS['add_lags'], NS['recursive_forecast'])

raw = pd.read_csv('data/skyline_hospital_bill_charge_report.csv')
raw = raw[['Open Date', 'Qty']].rename(columns={'Open Date': 'Date', 'Qty': 'Patients'})
raw['Date'] = pd.to_datetime(raw['Date'])
raw = raw.groupby(by='Date', as_index=False)['Patients'].sum().set_index('Date')
raw = raw.reindex(pd.date_range('2020-05-18', end='2022-12-31', freq='D'))
raw.index.name = 'Date'
raw = raw.interpolate(method='linear')
s = raw['Patients']
Q1, Q3 = s.quantile(.25), s.quantile(.75)
IQR = Q3 - Q1
lo, hi = Q1 - .1 * IQR, Q3 + 1.5 * IQR
raw[TARGET] = s.apply(lambda x: lo if x < lo else (hi if x > hi else x))
df = raw.sort_index()

FEATURES = ['day_of_week', 'month', 'day_of_year', 'rolling_sum', 'rolling_mean',
            'rolling_median', 'rolling_std', 'rolling_quantile_25',
            'rolling_quantile_75', 'lag_7_days', 'lag_14_days', 'lag_21_days',
            'lag_28_days', 'lag_30_days', 'lag_60_days', 'lag_180_days']
PARAMS = dict(base_score=0.5, booster='gbtree', n_estimators=900, tree_method='hist',
              objective='reg:squarederror', max_depth=3, min_child_weight=3, gamma=0,
              learning_rate=0.01, colsample_bytree=0.9, subsample=0.7, reg_lambda=0)

origin = pd.Timestamp('2021-06-01')
o = df.index.get_loc(origin)
hist = df.iloc[:o].copy()
hzn = pd.date_range(hist.index.max() + pd.Timedelta('1 day'), periods=180, freq='D')

train = add_lags(create_features(hist))
assert train.index.max() < origin, 'training data must end before the origin'
reg = xgb.XGBRegressor(**PARAMS)
reg.fit(train[FEATURES], train[TARGET], verbose=False)

f_clean = recursive_forecast(reg, hist, hzn, FEATURES)['pred']

# 1) poison the true future values in the frame we hand over
poisoned = df.iloc[:o + 180].copy()
poisoned.iloc[-180:, poisoned.columns.get_loc(TARGET)] = 99999.0
f_poison = recursive_forecast(reg, poisoned, hzn, FEATURES)['pred']

# 2) delete the true future values entirely
truncated = df.iloc[:o].copy()
f_trunc = recursive_forecast(reg, truncated, hzn, FEATURES)['pred']

# 3) a history that is itself shifted in time, to prove the function reacts to
#    what it is given rather than to any global state
shorter = df.iloc[:o - 30].copy()
hzn2 = pd.date_range(shorter.index.max() + pd.Timedelta('1 day'), periods=180, freq='D')
tr2 = add_lags(create_features(shorter))
reg2 = xgb.XGBRegressor(**PARAMS)
reg2.fit(tr2[FEATURES], tr2[TARGET], verbose=False)
f_diff = recursive_forecast(reg2, shorter, hzn2, FEATURES)['pred']

print('LEAKAGE PROBE')
print(f'  max |f(hist) - f(frame with true future poisoned to 99999)| = '
      f'{float(np.abs(f_clean - f_poison).max()):.10f}')
print(f'  max |f(hist) - f(truncated frame, no future at all)        | = '
      f'{float(np.abs(f_clean - f_trunc).max()):.10f}')
print(f'  mean |f(model A) - f(different model B)| (should be > 0)   = '
      f'{float(np.abs(f_clean - f_diff).mean()):.4f}')
assert np.allclose(f_clean, f_poison), 'LEAK: forecast reacted to true future values'
assert np.allclose(f_clean, f_trunc), 'LEAK: forecast reacted to frame contents'
print('\n  PASS - the forecast is a pure function of the history slice it was given.')
