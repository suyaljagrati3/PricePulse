"""End-to-end batch workflow orchestration."""

from pathlib import Path
import numpy as np
import pandas as pd

from .config import CV_FOLDS, FORECAST_MONTHS, N_RANDOM, OUTLIER_METHOD, USE_GRIDSEARCH, USE_PROPHET, USE_RANDOMSEARCH
from .data import build_forecast_targets
from .metrics import calculate_metrics, tscv_splits
from .models import forecast_prophet, forecast_sarima, gridsearch_sarima, randomsearch_sarima
from .reporting import plot_cv_summary


def _cross_validate_sarima(values):
    scores = []
    for train, validation in tscv_splits(len(values), FORECAST_MONTHS, CV_FOLDS, max(12, int(len(values) * .4))):
        try:
            prediction, *_ = forecast_sarima(values[train], len(validation))
            scores.append(calculate_metrics(values[validation], prediction)["MAPE"])
        except Exception as error:
            print(f"    CV fold failed: {error}")
    return float(np.mean(scores)) if scores else None


def _evaluate(name, forecast, train, test, full_values, results, predictions, **metadata):
    try:
        prediction, *_ = forecast(train)
        score = calculate_metrics(test, prediction, name)
        if score["MAPE"] <= 500:
            score.update({"Model": name, **metadata})
            results.append(score)
            predictions[name] = forecast(full_values)[0]
    except Exception as error:
        print(f"    {name} failed: {error}")


def run_forecasts(data_dir="datasets", show_plots=True):
    """Train enabled forecasting models and return accuracy and prediction data."""
    targets = build_forecast_targets(data_dir, OUTLIER_METHOD)
    accuracy, best_models, cv_results, forecasts = [], {}, [], {}
    for target in targets:
        name, frame = target["col"], target["series"]
        values, dates = frame[name].to_numpy(), frame["Date"]
        if len(values) <= FORECAST_MONTHS + 4:
            print(f"Skipping {name}: insufficient observations")
            continue
        train, test = values[:-FORECAST_MONTHS], values[-FORECAST_MONTHS:]
        train_dates = dates.iloc[:-FORECAST_MONTHS]
        model_results, model_forecasts = [], {}
        print(f"\nSeries: {name} (train={len(train)}, test={len(test)})")
        cv_score = _cross_validate_sarima(train)
        if cv_score is not None:
            cv_results.append({"Series": name, "CV_MAPE": round(cv_score, 2), "Folds": CV_FOLDS})
        _evaluate("SARIMA_Default", lambda series: forecast_sarima(series, FORECAST_MONTHS), train, test, values, model_results, model_forecasts, Series=name)
        if USE_GRIDSEARCH:
            _evaluate("SARIMA_GridSearch", lambda series: gridsearch_sarima(series, FORECAST_MONTHS), train, test, values, model_results, model_forecasts, Series=name)
        if USE_RANDOMSEARCH:
            _evaluate("SARIMA_RandomSearch", lambda series: randomsearch_sarima(series, FORECAST_MONTHS, N_RANDOM), train, test, values, model_results, model_forecasts, Series=name)
        if USE_PROPHET:
            _evaluate("Prophet", lambda series: forecast_prophet(train_dates if len(series) == len(train) else dates, series, FORECAST_MONTHS, series_name=name), train, test, values, model_results, model_forecasts, Series=name)
        accuracy.extend(model_results)
        forecasts[name] = model_forecasts
        if model_results:
            best_models[name] = min(model_results, key=lambda item: item["MAPE"])
    if show_plots:
        import matplotlib.pyplot as plt
        plot_cv_summary(cv_results, CV_FOLDS, plt)
    accuracy_frame = pd.DataFrame(accuracy)
    if not accuracy_frame.empty:
        accuracy_frame.to_csv(Path("model_accuracy_all.csv"), index=False)
    return {"targets": targets, "accuracy": accuracy_frame, "best_models": best_models, "cv_results": cv_results, "forecasts": forecasts}
