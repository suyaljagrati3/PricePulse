"""Central configuration for the batch forecasting workflow."""

FORECAST_MONTHS = 6
USE_PROPHET = True
USE_LSTM = True
USE_GRIDSEARCH = True
USE_RANDOMSEARCH = True
USE_XGBOOST = True
USE_RF = True
USE_HYBRID = True
N_RANDOM = 15
OUTLIER_METHOD = "iqr"
SCALER_TYPE = "minmax"
CV_FOLDS = 3
