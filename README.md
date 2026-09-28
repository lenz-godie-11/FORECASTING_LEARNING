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


- Evaluated the model with performance metrics across all the folds with root mean squared error (RMSE), mean absolute percentage error (MAPE), and symmetric mean absolute percentage error (sMAPE) with a 11% margin of error
- Plotted feature importance



- Generated future patient arrival forecasts

## Conclusion

- Real world time series data can be extremely dirty. I needed to do things like:
	- Dropping of irrelevant columns
	- Renaming of columns
	-  Merging of duplicate dates
	- Missing value imputation
	- Outlier analysis
- The future patient arrival predictions were able to attain an 11% margin of error when tested on the testing set.
- Time lag features are crucial for time series forecasting. For this project, the 7-day time lag was the most effective feature in helping the XGBoost model's forecasting ability.
