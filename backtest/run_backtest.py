"""
Historical walk-forward backtest of the recursive 180-day forecasting pipeline.

This script does NOT modify Solution_2.ipynb or the production forecast. It
imports `create_features`, `add_lags` and `recursive_forecast` straight out of
the executed notebook, so the backtest provably exercises the same code the
production forecast uses, with zero risk of drift.

Leakage discipline
------------------
At each forecast origin the model is fitted on a slice of the series that ends
strictly before the origin. The recursive loop is then handed ONLY that slice,
so the actual future values cannot enter the forecast features; they are joined
back in only afterwards, for scoring.

Baselines are made recursive too, so all methods face the same information set.
"""
import json
import re
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

NB = 'Solution_2.ipynb'
HORIZON = 180
OUT = 'backtest'

# ---------------------------------------------------------------------------
# 1. Pull the production functions out of the notebook
# ---------------------------------------------------------------------------
with open(NB) as f:
    nb = json.load(f)

sources = [''.join(c['source']) for c in nb['cells']]


def grab(defname):
    """Return the full source of the cell that defines `defname`."""
    pat = re.compile(rf'^def\s+{defname}\s*\(', re.M)
    for src in sources:
        if pat.search(src):
            return src
    raise LookupError(f'{defname} not found in {NB}')


# The notebook cells that hold these definitions also end with a top-level call
# such as `df = create_features(df)`. Provide a throwaway frame so exec'ing the
# cell defines the function and then harmlessly re-runs the call on the dummy.
_dummy = pd.DataFrame({'Patients': [1.0, 2.0, 3.0]},
                      index=pd.date_range('2020-01-01', periods=3, freq='D'))
NS = {'pd': pd, 'np': np, 'xgb': xgb, 'df': _dummy, 'label': None}
for fn in ('create_features', 'add_lags', 'recursive_forecast'):
    exec(compile(grab(fn), fn, 'exec'), NS)
create_features, add_lags, recursive_forecast = (
    NS['create_features'], NS['add_lags'], NS['recursive_forecast'])
print('loaded from notebook: create_features, add_lags, recursive_forecast')

# ---------------------------------------------------------------------------
# 2. Rebuild the history exactly as the notebook does, with verification
# ---------------------------------------------------------------------------
raw = pd.read_csv('data/skyline_hospital_bill_charge_report.csv')
raw = raw[['Open Date', 'Qty']].rename(columns={'Open Date': 'Date', 'Qty': 'Patients'})
raw['Date'] = pd.to_datetime(raw['Date'])
raw = raw.groupby(by='Date', as_index=False)['Patients'].sum().set_index('Date')
raw = raw.reindex(pd.date_range(start='2020-05-18', end='2022-12-31', freq='D'))
raw.index.name = 'Date'
raw = raw.interpolate(method='linear')

s = raw['Patients']
Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 0.1 * IQR, Q3 + 1.5 * IQR
raw['Patients'] = s.apply(
    lambda x: lower_bound if x < lower_bound
    else (upper_bound if x > upper_bound else x))

# verify against the artefacts the notebook produced
committed = pd.read_parquet('data/patient_arrivals.parquet')
assert len(raw) == 958, len(raw)
assert len(committed) == 856
assert int(committed['Patients'].sum()) == int(
    pd.read_csv('data/skyline_hospital_bill_charge_report.csv')['Qty'].sum())
assert np.isclose(lower_bound, 106.2) and np.isclose(upper_bound, 309.0), \
    (lower_bound, upper_bound)
print(f'history {len(raw)} rows  {raw.index.min().date()} -> {raw.index.max().date()}')
print(f'winsor bounds [{lower_bound:.1f}, {upper_bound:.1f}]  (match notebook output)')

df = raw.sort_index()
TARGET = 'Patients'
FEATURES = ['day_of_week', 'month', 'day_of_year', 'rolling_sum', 'rolling_mean',
            'rolling_median', 'rolling_std', 'rolling_quantile_25',
            'rolling_quantile_75', 'lag_7_days', 'lag_14_days', 'lag_21_days',
            'lag_28_days', 'lag_30_days', 'lag_60_days', 'lag_180_days']

# same estimator config as notebook cell 85
PARAMS = dict(base_score=0.5, booster='gbtree', n_estimators=900, tree_method='hist',
              objective='reg:squarederror', max_depth=3, min_child_weight=3, gamma=0,
              learning_rate=0.01, colsample_bytree=0.9, subsample=0.7, reg_lambda=0)

# ---------------------------------------------------------------------------
# 3. Metrics (RMSE, MAE, MAPE, sMAPE)
# ---------------------------------------------------------------------------
def rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(p)) ** 2)))


def mae(a, p):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(p))))


def mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.mean(np.abs((a - p) / a)) * 100)


def smape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(a - p) / (np.abs(a) + np.abs(p))) * 100)


def score(actual, pred):
    return dict(RMSE=rmse(actual, pred), MAE=mae(actual, pred),
                MAPE=mape(actual, pred), sMAPE=smape(actual, pred))


# ---------------------------------------------------------------------------
# 4. Recursive baselines, given exactly the same information as the model
# ---------------------------------------------------------------------------
def baseline_seasonal_naive(history, horizon, season=7):
    """pred[t] = value 7 days earlier, chained through its own predictions."""
    ext = history[TARGET].astype(float).to_dict()
    out = {}
    for d in horizon:
        out[d] = ext.get(d - pd.Timedelta(days=season))
        ext[d] = out[d]
    return pd.Series(out).reindex(horizon)


def baseline_weekday_mean(history, horizon, weeks=8):
    """mean of the same weekday over the last `weeks` observations, chained."""
    ext = history[TARGET].astype(float).to_dict()
    out = {}
    for d in horizon:
        same_wd = [ext[d - pd.Timedelta(days=7 * k)] for k in range(1, weeks + 1)]
        same_wd = [v for v in same_wd if v is not None and not pd.isna(v)]
        out[d] = float(np.mean(same_wd)) if same_wd else np.nan
        ext[d] = out[d]
    return pd.Series(out).reindex(horizon)


def baseline_last_value(history, horizon):
    """constant forecast at the final observed value."""
    return pd.Series(float(history[TARGET].iloc[-1]), index=horizon)


# ---------------------------------------------------------------------------
# 5. Walk-forward origins
# ---------------------------------------------------------------------------
MIN_TRAIN_DAYS = 365
last_date = df.index.max()
max_origin = last_date - pd.Timedelta(days=HORIZON)
min_origin = df.index.min() + pd.Timedelta(days=MIN_TRAIN_DAYS)

origins = pd.date_range(start='2021-06-01', end=max_origin, freq='MS')
print(f'\nvalid origin window: {min_origin.date()} .. {max_origin.date()}')
print(f'using {len(origins)} monthly origins: '
      f'{origins[0].date()} .. {origins[-1].date()}')

# ---------------------------------------------------------------------------
# 6. Run the backtest
# ---------------------------------------------------------------------------
rows, per_origin, per_horizon = [], {}, []
BIN_EDGES = [(1, 30), (31, 60), (61, 90), (91, 120), (121, 150), (151, 180)]

for origin in origins:
    # Convention: the forecast ORIGIN is the last day of the training data.
    # The horizon is the 180 days that follow it.
    o_pos = df.index.get_loc(origin)
    history = df.iloc[:o_pos + 1].copy()      # ends ON the origin
    horizon = pd.date_range(start=origin + pd.Timedelta(days=1),
                            periods=HORIZON, freq='D')

    assert history.index.max() == origin, 'history must end exactly on the origin'
    assert horizon.min() == origin + pd.Timedelta(days=1), 'gap between origin and horizon'
    assert horizon.max() <= df.index.max(), 'horizon runs past the data'
    assert history.index.intersection(horizon).empty, 'history and horizon overlap'

    # ---- fit on the past only ----
    train = add_lags(create_features(history))
    reg = xgb.XGBRegressor(**PARAMS)
    reg.fit(train[FEATURES], train[TARGET], verbose=False)

    # ---- recursive forecast, handed ONLY the past ----
    fc = recursive_forecast(reg, history, horizon, FEATURES)
    assert not fc[FEATURES].isna().any(axis=None), f'{origin.date()}: NaN features'
    pred = fc['pred']

    # ---- only NOW bring in the actuals, for scoring ----
    actual = df.loc[horizon, TARGET]
    assert len(actual) == HORIZON and not actual.isna().any()

    sn = baseline_seasonal_naive(history, horizon)
    wm = baseline_weekday_mean(history, horizon)
    lv = baseline_last_value(history, horizon)

    m = score(actual, pred)
    rows.append(dict(Origin=origin.date(), TrainDays=len(history),
                     TrainStart=history.index.min().date(), TrainEnd=origin.date(),
                     HorizonStart=horizon.min().date(), HorizonEnd=horizon.max().date(),
                     **{f'XGB_{k}': v for k, v in m.items()},
                     **{f'SeasNaive_{k}': v for k, v in score(actual, sn).items()},
                     **{f'Weekday8_{k}': v for k, v in score(actual, wm).items()},
                     **{f'LastValue_{k}': v for k, v in score(actual, lv).items()}))
    per_origin[origin] = dict(actual=actual, xgb=pred, sn=sn, wm=wm, lv=lv)

    # per-day error by horizon band, for every method
    methods = {'XGB': pred, 'SeasNaive': sn, 'Weekday8': wm, 'LastValue': lv}
    for mname, mpred in methods.items():
        e = pd.DataFrame({'day': np.arange(1, HORIZON + 1),
                          'abs_pct': np.abs((actual.to_numpy() - mpred.to_numpy())
                                            / actual.to_numpy()) * 100,
                          'abs': np.abs(actual.to_numpy() - mpred.to_numpy())})
        for lo, hi in BIN_EDGES:
            sel = e[(e['day'] >= lo) & (e['day'] <= hi)]
            per_horizon.append(dict(Origin=origin.date(), Method=mname,
                                    band=f'{lo}-{hi}', MAPE=sel['abs_pct'].mean(),
                                    MAE=sel['abs'].mean(),
                                    RMSE=float(np.sqrt((sel['abs'] ** 2).mean()))))

    print(f'  {origin.date()}  train={len(history):>3}d  '
          f'RMSE {m["RMSE"]:6.2f}  MAE {m["MAE"]:6.2f}  MAPE {m["MAPE"]:5.2f}%  '
          f'sMAPE {m["sMAPE"]:5.2f}%')

results = pd.DataFrame(rows)
import os
os.makedirs(OUT, exist_ok=True)
results.to_csv(f'{OUT}/backtest_results.csv', index=False)
pd.DataFrame(per_horizon).to_csv(f'{OUT}/backtest_by_horizon.csv', index=False)
pd.to_pickle(per_origin, f'{OUT}/per_origin.pkl')

# ---------------------------------------------------------------------------
# 7. Reporting tables
# ---------------------------------------------------------------------------
pd.set_option('display.width', 250, 'display.max_columns', 50)
cols = ['Origin', 'TrainDays', 'RMSE', 'MAE', 'MAPE', 'sMAPE']
print('\n' + '=' * 100)
print('XGBOOST RECURSIVE — 180-DAY HORIZON, PER ORIGIN')
print('=' * 100)
print(results[['Origin', 'TrainDays', 'XGB_RMSE', 'XGB_MAE', 'XGB_MAPE', 'XGB_sMAPE']]
      .rename(columns={c: c.replace('XGB_', '') for c in
                       ['XGB_RMSE', 'XGB_MAE', 'XGB_MAPE', 'XGB_sMAPE']})
      .round(2).to_string(index=False))

print('\n' + '=' * 100)
print('OVERALL ACROSS ORIGINS')
print('=' * 100)
summary = []
for name, tag in [('XGBoost recursive', 'XGB'), ('Seasonal naive (7d)', 'SeasNaive'),
                  ('Weekday mean (8w)', 'Weekday8'), ('Last value', 'LastValue')]:
    row = {'Method': name}
    for metric in ['RMSE', 'MAE', 'MAPE', 'sMAPE']:
        row[f'{metric} mean'] = results[f'{tag}_{metric}'].mean()
        row[f'{metric} median'] = results[f'{tag}_{metric}'].median()
    summary.append(row)
summary = pd.DataFrame(summary)
print(summary.round(2).to_string(index=False))
summary.to_csv(f'{OUT}/backtest_summary.csv', index=False)

# ---------------------------------------------------------------------------
# 8. Error vs horizon length
# ---------------------------------------------------------------------------
ph = pd.DataFrame(per_horizon)
order = [f'{a}-{b}' for a, b in BIN_EDGES]
agg = ph.groupby(['Method', 'band'])[['RMSE', 'MAE', 'MAPE']].mean()
agg = agg.reindex(pd.MultiIndex.from_product([['XGB', 'SeasNaive', 'Weekday8',
                                              'LastValue'], order],
                                             names=['Method', 'band']))
print('\n' + '=' * 100)
print('ERROR BY HORIZON BAND (mean across all 14 origins)')
print('=' * 100)
print(agg.round(2).to_string())

print('\nXGBoost per-origin MAPE by band:')
piv = ph[ph.Method == 'XGB'].pivot(index='Origin', columns='band', values='MAPE').reindex(
    columns=order)
print(piv.round(2).to_string())
agg.reset_index().to_csv(f'{OUT}/backtest_horizon_bands.csv', index=False)
piv.to_csv(f'{OUT}/backtest_horizon_bands_xgb_by_origin.csv')

# how much does the forecast flatten out relative to reality?
print('\n' + '=' * 100)
print('FORECAST DISPERSION vs REALITY (mean across origins)')
print('=' * 100)
disp = []
for origin, d in per_origin.items():
    a = d['actual'].to_numpy()
    for mname, key in [('XGB', 'xgb'), ('SeasNaive', 'sn'), ('Weekday8', 'wm')]:
        p = d[key].to_numpy()
        disp.append(dict(Origin=origin.date(), Method=mname,
                         actual_std=a.std(), pred_std=p.std(),
                         std_ratio=p.std() / a.std(),
                         actual_mean=a.mean(), pred_mean=p.mean(),
                         bias_pct=100 * (p.mean() / a.mean() - 1)))
disp = pd.DataFrame(disp)
print(disp.groupby('Method')[['actual_std', 'pred_std', 'std_ratio',
                              'bias_pct']].mean().round(2).to_string())
disp.to_csv(f'{OUT}/backtest_dispersion.csv', index=False)

json.dump({'origins': [str(o.date()) for o in origins],
           'summary': summary.to_dict('records'),
           'bands': agg.reset_index().to_dict('records')},
          open(f'{OUT}/backtest_report.json', 'w'), indent=2, default=str)

print(f'\nresults written to {OUT}/')
