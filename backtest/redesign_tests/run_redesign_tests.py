"""
Redesign tests, run through the same walk-forward backtest harness as
backtest/run_backtest.py: same 14 origins, same 180-day horizon, same leakage
guards, same metrics, same horizon bands.

New methods under test
  * direct multi-horizon XGBoost (180 models per origin)
  * damped-trend / ETS / Holt / Theta classical forecasters

Carried-over comparison methods (unchanged implementations)
  * recursive XGBoost, weekday mean (8w), seasonal naive (7d), last value

No features, hyperparameters, data or notebook cells are modified.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

from multiprocessing import Pool

from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from statsmodels.tsa.forecasting.theta import ThetaModel

OUT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# method 3: classical damped-trend / ETS / Theta family
# ---------------------------------------------------------------------------
def _fit_ets(y, **kw):
    return ExponentialSmoothing(y, **kw).fit(optimized=True).forecast(C.HORIZON)


def classical_forecasts(history, horizon):
    """
    Each forecaster produces the whole 180-day path in one shot from the
    history. They are direct by construction, so no prediction is fed back in.
    """
    y = history[C.TARGET].astype(float)
    out = {}

    specs = [
        ('ETS_damped_seas7', 'ets',
         dict(trend='add', damped_trend=True, seasonal='add', seasonal_periods=7)),
        ('ETS_trend_seas7', 'ets',
         dict(trend='add', damped_trend=False, seasonal='add', seasonal_periods=7)),
        ('ETS_seasonal_only', 'ets',
         dict(trend=None, damped_trend=False, seasonal='add', seasonal_periods=7)),
        ('Holt_damped', 'holt', dict(damped_trend=True)),
        ('Holt_plain', 'holt', dict(damped_trend=False)),
        ('Theta_p7', 'theta', dict(period=7)),
        ('Theta_p30', 'theta', dict(period=30)),
    ]

    for name, kind, kw in specs:
        try:
            if kind == 'ets':
                f = _fit_ets(y, **kw)
            elif kind == 'holt':
                f = Holt(y, **kw).fit().forecast(C.HORIZON)
            else:
                f = ThetaModel(y, **kw).fit().forecast(C.HORIZON)
            f = np.asarray(f).ravel()
            if len(f) < C.HORIZON:                    # guard against short output
                f = np.concatenate([f, np.repeat(f[-1], C.HORIZON - len(f))])
            out[name] = pd.Series(f[:C.HORIZON], index=horizon)
        except Exception as exc:                      # record, never silently skip
            out[name] = pd.Series(np.nan, index=horizon)
            print(f'    WARNING {name} failed at this origin: {type(exc).__name__}: {exc}')
    return out


# ---------------------------------------------------------------------------
# worker: everything for one origin
# ---------------------------------------------------------------------------
def run_one_origin(origin_str):
    origin = pd.Timestamp(origin_str)
    history, horizon = C.origin_slice(origin)          # leakage guards live here

    res = {'Origin': origin.date(), 'TrainDays': len(history),
           'HorizonStart': horizon.min().date(), 'HorizonEnd': horizon.max().date()}

    preds = {}

    preds['XGB_recursive'] = C.forecast_recursive_xgb(history, horizon)

    direct, diag = C.forecast_direct_xgb(history, horizon)
    preds['XGB_direct'] = direct
    diag['Origin'] = origin.date()

    for name, s in classical_forecasts(history, horizon).items():
        preds[name] = s

    preds['SeasNaive'] = C.forecast_seasonal_naive(history, horizon)
    preds['Weekday8'] = C.forecast_weekday_mean(history, horizon)
    preds['LastValue'] = C.forecast_last_value(history, horizon)

    # ---- actuals enter ONLY now, for scoring ----
    actual = C.DF.loc[horizon, C.TARGET]
    assert len(actual) == C.HORIZON and not actual.isna().any()

    per_origin = {}
    for name, p in preds.items():
        assert len(p) == C.HORIZON, f'{name}: {len(p)} predictions'
        assert p.index.equals(horizon), f'{name}: wrong dates'
        s = C.score(actual, p)
        res.update({f'{name}_{k}': v for k, v in s.items()})
        per_origin[name] = p

    return res, per_origin, actual, diag


def _init():
    pass


if __name__ == '__main__':
    t0 = time.time()
    origins = C.get_origins()
    print(f'origins: {origins[0].date()} .. {origins[-1].date()}  (n={len(origins)})',
          flush=True)
    print(f'nthread={C.PARAMS["nthread"]}  workers={min(7, os.cpu_count() or 1)}',
          flush=True)
    print('direct multi-horizon = 180 model fits per origin\n', flush=True)

    # CHECKPOINT: each completed origin is flushed to disk immediately, so an
    # interrupted run loses at most one origin instead of everything.
    CKPT = f'{OUT}/checkpoint_per_origin.csv'
    CKPT_PKL = f'{OUT}/checkpoint_per_origin.pkl'
    if os.path.exists(CKPT):
        os.remove(CKPT)
    done = {}

    results, per_origin, actuals, diags = [], {}, {}, []
    with Pool(processes=min(7, os.cpu_count() or 1)) as pool:
        for res, po, act, dg in pool.imap_unordered(
                run_one_origin, [str(o.date()) for o in origins]):
            results.append(res)
            per_origin[pd.Timestamp(res['Origin'])] = po
            actuals[pd.Timestamp(res['Origin'])] = act
            diags.append(dg)
            done[res['Origin']] = (res, po, act, dg)

            x = res
            print(f'  [{len(results):>2}/{len(origins)}] {x["Origin"]}  '
                  f'train={x["TrainDays"]:>3}d  '
                  f'recursive MAPE {x["XGB_recursive_MAPE"]:6.2f}%  '
                  f'direct MAPE {x["XGB_direct_MAPE"]:6.2f}%  '
                  f'ETS_damped {x["ETS_damped_seas7_MAPE"]:6.2f}%  '
                  f'wmean {x["Weekday8_MAPE"]:6.2f}%  '
                  f'[{time.time() - t0:6.0f}s]', flush=True)

            # ---- flush checkpoint ----
            pd.DataFrame([r for r, *_ in done.values()]).sort_values('Origin') \
                .to_csv(CKPT, index=False)
            pd.to_pickle({'preds': {pd.Timestamp(k): v[1] for k, v in done.items()},
                          'actual': {pd.Timestamp(k): v[2] for k, v in done.items()}},
                         CKPT_PKL)
            pd.concat(diags, ignore_index=True).to_csv(
                f'{OUT}/checkpoint_direct_train_sizes.csv', index=False)

    print(f'\nall origins complete in {time.time() - t0:.0f}s', flush=True)

    results = pd.DataFrame(results).sort_values('Origin').reset_index(drop=True)
    results.to_csv(f'{OUT}/redesign_results.csv', index=False)
    pd.concat(diags, ignore_index=True).to_csv(f'{OUT}/direct_train_sizes.csv', index=False)
    pd.to_pickle({'preds': per_origin, 'actual': actuals},
                 f'{OUT}/redesign_per_origin.pkl')

    methods = ['XGB_direct', 'XGB_recursive', 'ETS_damped_seas7', 'ETS_trend_seas7',
               'ETS_seasonal_only', 'Holt_damped', 'Holt_plain', 'Theta_p7', 'Theta_p30',
               'Weekday8', 'SeasNaive', 'LastValue']

    pd.set_option('display.width', 250, 'display.max_columns', 60)
    print('\n' + '=' * 110)
    print('OVERALL — mean across 14 origins, 180-day horizon')
    print('=' * 110)
    summ = pd.DataFrame([
        dict(Method=m, RMSE=results[f'{m}_RMSE'].mean(), MAE=results[f'{m}_MAE'].mean(),
             MAPE=results[f'{m}_MAPE'].mean(), sMAPE=results[f'{m}_sMAPE'].mean(),
             RMSE_med=results[f'{m}_RMSE'].median(), MAPE_med=results[f'{m}_MAPE'].median())
        for m in methods]).sort_values('MAPE')
    print(summ.round(2).to_string(index=False))
    summ.to_csv(f'{OUT}/redesign_summary.csv', index=False)

    # ---- per-horizon bands ----
    recs = []
    for origin, po in per_origin.items():
        a = actuals[origin].to_numpy()
        for name, p in po.items():
            e = np.abs((a - p.to_numpy()) / a) * 100
            ae = np.abs(a - p.to_numpy())
            for lo, hi in C.BAND_EDGES:
                sl = slice(lo - 1, hi)
                recs.append(dict(Origin=origin.date(), Method=name,
                                 band=f'{lo}-{hi}',
                                 MAPE=e[sl].mean(), MAE=ae[sl].mean(),
                                 RMSE=float(np.sqrt((ae[sl] ** 2).mean()))))
    bands = pd.DataFrame(recs)
    order = [f'{a}-{b}' for a, b in C.BAND_EDGES]
    piv = bands.pivot_table(index='Method', columns='band', values='MAPE').reindex(methods)
    print('\n' + '=' * 110)
    print('MAPE BY HORIZON BAND (mean across origins)')
    print('=' * 110)
    print(piv.reindex(columns=order).round(2).to_string())
    bands.to_csv(f'{OUT}/redesign_bands.csv', index=False)
    piv.to_csv(f'{OUT}/redesign_bands_pivot.csv')

    print('\nwinning method per band:')
    for b in order:
        col = piv[b].dropna()
        if col.empty:
            continue
        print(f'  {b:>9} : {col.idxmin():20s} {col.min():6.2f}%   '
              f'(direct {piv.loc["XGB_direct", b]:.2f}%, '
              f'recursive {piv.loc["XGB_recursive", b]:.2f}%)')

    dt = pd.concat(diags, ignore_index=True)
    print('\ndirect multi-horizon effective training size (rows with NO NaN features):')
    print(dt.groupby('Origin')[['n_train_rows', 'n_train_complete']].agg(
        ['min', 'max']).round(0).to_string())
    print(f'\nelapsed {time.time() - t0:.0f}s')
