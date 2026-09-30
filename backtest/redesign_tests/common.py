"""
Shared harness for the redesign tests.

Deliberately reuses the production functions by importing them straight out of
Solution_2.ipynb, exactly as backtest/run_backtest.py does, so that every method
tested here is evaluated on the same features, the same model configuration and
the same data as the production forecast.

Nothing in this module writes to the notebook, the data, or the saved model.
"""
import json
import os
import re
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb

TARGET = 'Patients'

# repository root, so the module works from any working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..'))
NB = os.path.join(ROOT, 'Solution_2.ipynb')
CSV = os.path.join(ROOT, 'data', 'skyline_hospital_bill_charge_report.csv')
HORIZON = 180

FEATURES = ['day_of_week', 'month', 'day_of_year', 'rolling_sum', 'rolling_mean',
            'rolling_median', 'rolling_std', 'rolling_quantile_25',
            'rolling_quantile_75', 'lag_7_days', 'lag_14_days', 'lag_21_days',
            'lag_28_days', 'lag_30_days', 'lag_60_days', 'lag_180_days']

# identical to notebook cell 85 / cell 76, except for nthread.
# nthread=1 is a PERFORMANCE setting only: it controls how many CPU threads
# XGBoost uses to build trees. It does not affect the objective, the splits, the
# random number stream, or the resulting predictions. It is set to 1 here so
# that the multiprocessing pool can run origins in parallel without 7 workers
# each spawning their own thread pool on an 8-core box.
PARAMS = dict(base_score=0.5, booster='gbtree', n_estimators=900, tree_method='hist',
              objective='reg:squarederror', max_depth=3, min_child_weight=3, gamma=0,
              learning_rate=0.01, colsample_bytree=0.9, subsample=0.7, reg_lambda=0,
              nthread=1)

BAND_EDGES = [(1, 30), (31, 60), (61, 90), (91, 120), (121, 150), (151, 180)]


# ---------------------------------------------------------------------------
# production functions, imported from the notebook
# ---------------------------------------------------------------------------
def _load_production_functions():
    with open(NB) as f:
        nb = json.load(f)
    sources = [''.join(c['source']) for c in nb['cells']]

    def grab(name):
        pat = re.compile(rf'^def\s+{name}\s*\(', re.M)
        for s in sources:
            if pat.search(s):
                return s
        raise LookupError(f'{name} not found in {NB}')

    dummy = pd.DataFrame({TARGET: [1.0, 2.0, 3.0]},
                         index=pd.date_range('2020-01-01', periods=3, freq='D'))
    ns = {'pd': pd, 'np': np, 'xgb': xgb, 'df': dummy}
    for fn in ('create_features', 'add_lags', 'recursive_forecast'):
        exec(compile(grab(fn), fn, 'exec'), ns)
    return ns['create_features'], ns['add_lags'], ns['recursive_forecast']


create_features, add_lags, recursive_forecast = _load_production_functions()


# ---------------------------------------------------------------------------
# history, rebuilt exactly as the notebook builds it
# ---------------------------------------------------------------------------
def build_history():
    raw = pd.read_csv(CSV)
    raw = raw[['Open Date', 'Qty']].rename(columns={'Open Date': 'Date', 'Qty': TARGET})
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.groupby(by='Date', as_index=False)[TARGET].sum().set_index('Date')
    raw = raw.reindex(pd.date_range(start='2020-05-18', end='2022-12-31', freq='D'))
    raw.index.name = 'Date'
    raw = raw.interpolate(method='linear')
    s = raw[TARGET]
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 0.1 * IQR, Q3 + 1.5 * IQR
    raw[TARGET] = s.apply(lambda x: lo if x < lo else (hi if x > hi else x))
    assert len(raw) == 958
    assert np.isclose(lo, 106.2) and np.isclose(hi, 309.0)
    return raw.sort_index()


DF = build_history()


def get_origins():
    """The same 14 monthly origins used in backtest/run_backtest.py."""
    max_origin = DF.index.max() - pd.Timedelta(days=HORIZON)
    return pd.date_range(start='2021-06-01', end=max_origin, freq='MS')


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))


def mae(a, p):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


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
# slice for one origin, with the leakage guards
# ---------------------------------------------------------------------------
def origin_slice(origin):
    """Return (history, horizon) with hard leakage assertions."""
    o = DF.index.get_loc(origin)
    history = DF.iloc[:o + 1].copy()                    # ends ON the origin
    horizon = pd.date_range(start=origin + pd.Timedelta(days=1),
                            periods=HORIZON, freq='D')
    assert history.index.max() == origin
    assert horizon.min() == origin + pd.Timedelta(days=1)
    assert horizon.max() <= DF.index.max()
    assert history.index.intersection(horizon).empty
    return history, horizon


# ---------------------------------------------------------------------------
# method 1: recursive XGBoost (the current production approach)
# ---------------------------------------------------------------------------
def forecast_recursive_xgb(history, horizon):
    train = add_lags(create_features(history))
    reg = xgb.XGBRegressor(**PARAMS)
    reg.fit(train[FEATURES], train[TARGET], verbose=False)
    fc = recursive_forecast(reg, history, horizon, FEATURES)
    assert not fc[FEATURES].isna().any(axis=None)
    return fc['pred'].reindex(horizon)


# ---------------------------------------------------------------------------
# method 2: DIRECT multi-horizon XGBoost
# ---------------------------------------------------------------------------
def forecast_direct_xgb(history, horizon):
    """
    One model per horizon. Model h is trained to map the past-only features at
    date t to the value observed at t+h, then applied once at the origin.

    No prediction is ever fed back in, so there is no error accumulation and no
    distribution drift across the horizon. Features, target construction and
    hyperparameters are identical to the recursive pipeline.
    """
    feat = add_lags(create_features(history))
    origin = history.index.max()
    x_origin = feat.loc[[origin], FEATURES]

    # OPTIMISATION: `feat[FEATURES]` is built once and reused for all 180 fits.
    # The previous version called feat.copy() inside the loop, which copied every
    # column (including the intermediate feature columns) 180 times per origin.
    # Values, column order and per-column dtypes are unchanged; only the number
    # of redundant copies differs.
    feat_cols = feat[FEATURES]

    preds, n_rows, n_complete = {}, {}, {}
    for h in range(1, HORIZON + 1):
        y_h = history[TARGET].shift(-h)
        keep = y_h.notna()                      # a valid supervised target is required
        d = feat_cols.loc[keep]
        y = y_h.loc[keep]

        model = xgb.XGBRegressor(**PARAMS)
        model.fit(d, y, verbose=False)
        preds[horizon[h - 1]] = float(model.predict(x_origin)[0])
        n_rows[h] = len(d)
        n_complete[h] = int((~d.isna().any(axis=1)).sum())

    out = pd.Series(preds).reindex(horizon)
    diag = pd.DataFrame({'h': range(1, HORIZON + 1),
                         'n_train_rows': [n_rows[h] for h in range(1, HORIZON + 1)],
                         'n_train_complete': [n_complete[h] for h in range(1, HORIZON + 1)]})
    return out, diag


# ---------------------------------------------------------------------------
# baselines carried over from backtest/run_backtest.py, unchanged
# ---------------------------------------------------------------------------
def forecast_seasonal_naive(history, horizon, season=7):
    ext = history[TARGET].astype(float).to_dict()
    out = {}
    for d in horizon:
        out[d] = ext.get(d - pd.Timedelta(days=season))
        ext[d] = out[d]
    return pd.Series(out).reindex(horizon)


def forecast_weekday_mean(history, horizon, weeks=8):
    ext = history[TARGET].astype(float).to_dict()
    out = {}
    for d in horizon:
        vals = [ext[d - pd.Timedelta(days=7 * k)] for k in range(1, weeks + 1)]
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        out[d] = float(np.mean(vals)) if vals else np.nan
        ext[d] = out[d]
    return pd.Series(out).reindex(horizon)


def forecast_last_value(history, horizon):
    return pd.Series(float(history[TARGET].iloc[-1]), index=horizon)
