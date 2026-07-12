"""Evaluation, scaling, and time-series cross-validation utilities."""

import numpy as np
import pandas as pd


def get_scaler(scaler_type="minmax"):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    return StandardScaler() if scaler_type == "standard" else MinMaxScaler(feature_range=(0, 1))


def calculate_metrics(actual, predicted, label=""):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    size = min(len(actual), len(predicted))
    actual, predicted = actual[:size], predicted[:size]
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual):
        return {"MAE": 999, "RMSE": 999, "MAPE": 999, "SMAPE": 999}
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = min(np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100, 999.0)
    smape = min(np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100, 999.0)
    if label:
        print(f"    [{label}] MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}% SMAPE={smape:.2f}%")
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2), "SMAPE": round(smape, 2)}


def tscv_splits(size, forecast_months=6, n_folds=3, min_train=12):
    fold_size = max(forecast_months, (size - min_train) // n_folds)
    splits = []
    for fold in range(n_folds):
        validation_end = size - fold * fold_size
        validation_start = validation_end - fold_size
        if validation_start < min_train:
            break
        splits.append((list(range(validation_start)), list(range(validation_start, validation_end))))
    return list(reversed(splits))


def cross_validate_sarima(series, forecast_sarima, forecast_months=6, n_folds=3):
    values = np.asarray(series, dtype=float)
    splits = tscv_splits(len(values), forecast_months, n_folds, max(12, int(len(values) * 0.4)))
    if not splits:
        print(f"    TSCV skipped - not enough data (n={len(values)})")
        return None
    mapes = []
    print(f"    TSCV SARIMA ({len(splits)} folds)...")
    for number, (train, validation) in enumerate(splits, start=1):
        try:
            forecast, *_ = forecast_sarima(values[train], len(validation))
            mape = calculate_metrics(values[validation], forecast)["MAPE"]
            if mape < 500:
                mapes.append(mape)
                print(f"      Fold {number}: train={len(train)} val={len(validation)} MAPE={mape:.2f}%")
        except Exception as error:
            print(f"      Fold {number}: FAILED ({error})")
    return float(np.mean(mapes)) if mapes else None
