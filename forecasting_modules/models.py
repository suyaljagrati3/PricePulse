"""Forecasting model implementations with optional heavy dependencies imported lazily."""

import itertools
import random
import numpy as np
import pandas as pd

from .metrics import get_scaler


def get_d(series):
    from statsmodels.tsa.stattools import adfuller
    values = pd.Series(series).dropna()
    if len(values) < 4:
        return 1
    try:
        return 0 if adfuller(values, autolag="AIC")[1] < 0.05 else 1
    except Exception:
        return 1


def safe_periodicity(length):
    return next((period for period in (12, 6, 4, 3, 2) if length > 2 * period + 4), 0)


def fit_sarima_safe(series, order, seasonal_order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if seasonal_order[-1] <= 1:
        seasonal_order = (0, 0, 0, 0)
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    for method in ("lbfgs", "powell", "nm", "bfgs"):
        try:
            return model.fit(disp=False, maxiter=200, method=method)
        except Exception:
            continue
    return None


def forecast_sarima(series, steps):
    values = pd.Series(series).dropna().reset_index(drop=True)
    if len(values) < 4:
        raise RuntimeError(f"Series too short (n={len(values)})")
    difference, period = get_d(values), safe_periodicity(len(values))
    candidates = [((1, difference, 1), (1, 1, 1, period)), ((1, difference, 0), (0, 1, 0, period))] if period >= 2 else []
    candidates += [((1, difference, 1), (0, 0, 0, 0)), ((1, difference, 0), (0, 0, 0, 0))]
    fit = None
    for order, seasonal in candidates:
        fit = fit_sarima_safe(values, order, seasonal)
        if fit is not None:
            break
    if fit is None:
        raise RuntimeError("SARIMA failed all optimizers")
    forecast = fit.get_forecast(steps=steps)
    interval = forecast.conf_int(alpha=0.20)
    return forecast.predicted_mean, interval.iloc[:, 0], interval.iloc[:, 1], fit.aic


def _search_sarima(series, steps, combinations):
    values = pd.Series(series).dropna().reset_index(drop=True)
    difference, period = get_d(values), safe_periodicity(len(values))
    best = (np.inf, None, None)
    for p, q, seasonal_p, seasonal_q in combinations:
        if period < 2 and (seasonal_p or seasonal_q):
            continue
        seasonal = (seasonal_p, 1, seasonal_q, period) if period >= 2 else (0, 0, 0, 0)
        fit = fit_sarima_safe(values, (p, difference, q), seasonal)
        if fit is not None and fit.aic < best[0]:
            best = (fit.aic, fit, ((p, difference, q), seasonal))
    if best[1] is None:
        raise RuntimeError("No SARIMA model converged")
    forecast = best[1].get_forecast(steps=steps)
    interval = forecast.conf_int(alpha=0.20)
    return forecast.predicted_mean, interval.iloc[:, 0], interval.iloc[:, 1], best[0], best[2]


def gridsearch_sarima(series, steps):
    print("    Running GridSearch SARIMA...")
    return _search_sarima(series, steps, itertools.product((0, 1, 2), (0, 1, 2), (0, 1), (0, 1)))


def randomsearch_sarima(series, steps, n_iter=15):
    print(f"    Running RandomSearch SARIMA ({n_iter} iters)...")
    random.seed(42)
    candidates = [(random.choice((0, 1, 2)), random.choice((0, 1, 2)), random.choice((0, 1)), random.choice((0, 1))) for _ in range(n_iter)]
    return _search_sarima(series, steps, candidates)


def forecast_prophet(dates, values, steps, changepoint_prior=0.3, series_name=""):
    from prophet import Prophet
    volatile = {"Onion_Price", "Tomato_Price", "Potato_Price", "Veg_Price"}
    stable = {"Sugar", "Salt", "Milk", "Gram", "Tur", "Moong"}
    cp_scale, seasonality = (0.5, 20) if series_name in volatile else ((0.05, 5) if series_name in stable else (changepoint_prior, 15))
    train = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values}).dropna()
    model = Prophet(changepoint_prior_scale=cp_scale, seasonality_prior_scale=seasonality, yearly_seasonality=len(train) >= 24, weekly_seasonality=False, daily_seasonality=False, interval_width=0.80, n_changepoints=min(25, len(train) // 3))
    model.fit(train)
    forecast = model.predict(model.make_future_dataframe(periods=steps, freq="MS"))
    future = forecast[forecast["ds"] > train["ds"].max()]
    return future["yhat"].values, future["yhat_lower"].values, future["yhat_upper"].values
