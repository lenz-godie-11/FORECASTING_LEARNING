# Time Series Forecasting with XGBoost


This is a data analysis project that aimed to forecast future patient arrivals data on the outpatient unit . The repository is for the data analytics feature of the of our bachelor's degree thesis.


## Data Collection

- Hospital's IT department sent daily time series data of patient arrivals in a csv file via email.
- The time series data time frame was **from May 18, 2020 to December 31, 2022.**

## Environment Setup and Data Loading

The time series data was **loaded into Jupyter Lab using Python**. Relevant libraries for data analysis were also imported:

- `pandas`: for Data Management and Data Manipulation
- `numpy`: for numerical and scientific computing
- `matplotlib`, `seaborn`, `plotly`: for data visualization
- `sklearn.metrics` (scikit-learn): for model evaluation metrics such as RMSE, MAPE, and SMAPE
- `xgboost.XGBRegressor`: a gradient boosting type decision tree machine learning algorithm used for time series forecasting
- `TimeSeriesSplit`: for time series cross-validation

## Data Cleaning

The data cleaning methods used in here include:

- Subsetting columns needed for time series analysis
- Renaming DataFrame columns for better clarity



-  Merging duplicate dates with corresponding patient arrival counts


- Missing value imputation



- Outlier analysis

## Feature Engineering

- Added time series features (e.g., day of week, day of year, month, etc.)
- Added rolling statistics and quantiles

## Time Lags Creation

- Added lags in time to let the XGBoost Regressor model learn historical values and the temporal structure of the data.

## Time Series Cross-Validation

- Made use of `TimeSeriesSplit` to apply Sliding Window Cross-Valid
- Extracted the features and the target variable.
- Created training and testing sets


## Model Fitting, Forecasting, and Evaluation

- Fit the XGBoost model on the training set
- Validated on the testing set and plotted the results


- Evaluated the model with performance metrics across all the folds with root mean squared error (RMSE), mean absolute percentage error (MAPE), and symmetric mean absolute percentage error (sMAPE)
- Plotted feature importance

### Production forecasting: Theta(period=7), 90-day horizon

The XGBoost work above is retained as the machine-learning analysis. The
**production forecast is produced by Theta with a seasonal period of 7**, over a
**90-day horizon**.

This decision is backed by a walk-forward backtest over 14 historical forecast
origins (2021-06-01 to 2022-07-01), each forecasting the following 180 days from
data available strictly before the origin. Full code and results are in
`backtest/` and `backtest/redesign_tests/`.

Mean 180-day performance across those origins:

| Method | RMSE | MAPE | sMAPE |
|---|---|---|---|
| **Theta (period 7)** | **37.35** | **17.98%** | 17.90% |
| Weekday mean (8 weeks) | 39.16 | 18.27% | 18.41% |
| Seasonal naive (7 days) | 42.97 | 19.98% | 20.13% |
| XGBoost direct (multi-horizon) | 46.89 | 21.18% | 22.90% |
| XGBoost recursive | 51.27 | 22.68% | 25.33% |
| Last value | 51.60 | 24.41% | 26.19% |

Findings:

- The recursive XGBoost forecaster was the **second-worst competent method**,
  beaten significantly by Theta (Wilcoxon p = 0.013) and by the seasonal naive
  baseline (p = 0.002).
- **No XGBoost variant won any horizon band.** Direct multi-horizon improved on
  recursive (-1.50 pp, 11/14 origins) but still lost to the weekday mean
  (p = 0.042). With 958 observations and strong weekly seasonality, a
  three-parameter Theta model matched a 900-tree ensemble.
- Theta's advantage over the weekday mean was small and not statistically
  significant (p = 0.626); its advantage over seasonal naive was significant
  (13/14 origins, p = 0.002).
- The horizon was cut from 180 to 90 days because **no method was reliable across
  a full 180 days**. Theta first exceeds a 20% MAPE around day 141.

### Known limitation

The Theta forecast can under-shoot at level shifts. In the backtest, the worst
single origin produced a -33.9% bias. The current production forecast averages
127.90 patients/day against a full-history mean of 158.60 (-19.4%) and a
last-90-day mean of 181.41 (-29.5%). For capacity planning, the forecast should
be read as a seasonal baseline, not a surge detector.

## Conclusion

- Real world time series data can be extremely dirty. I needed to do things like:
	- Dropping of irrelevant columns
	- Renaming of columns
	-  Merging of duplicate dates
	- Missing value imputation
	- Outlier analysis
- Time lag features are crucial for time series forecasting. For this project, the 7-day time lag was the most effective feature in helping the XGBoost model's forecasting ability.
- **A walk-forward backtest is essential before trusting a long-horizon forecast.** The original ~11% figure came from cross-validation that included a fold trained on 5 rows and from a rolling-window feature that contained the target itself. Once both were corrected, the honest 180-day error was 22.68% - and the model's advantage over a simple seasonal baseline was negative.
- **On this dataset, a classical Theta model beat a 900-tree gradient boosting ensemble at every horizon.** With limited observations and dominant weekly seasonality, added model complexity did not translate into better long-range forecasts.
