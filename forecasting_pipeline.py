import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import warnings
import itertools
import random

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
WORK_DIR         = os.getcwd()
FORECAST_MONTHS  = 6
USE_PROPHET      = True
USE_LSTM         = True
USE_GRIDSEARCH   = True
USE_RANDOMSEARCH = True
USE_XGBOOST      = True
USE_RF           = True
USE_HYBRID       = True
N_RANDOM         = 15
OUTLIER_METHOD   = "iqr"
SCALER_TYPE      = "minmax"
CV_FOLDS         = 3

os.chdir(WORK_DIR)

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
PALETTE = ["#2563EB","#DC2626","#16A34A","#D97706","#7C3AED","#0891B2",
           "#DB2777","#059669","#B45309","#6D28D9","#0369A1","#BE123C",
           "#047857","#92400E","#4338CA","#0F766E","#BE185D","#15803D"]

# ── Themed Dashboard Groups ────────────────────────────────────────
DASHBOARD_GROUPS = {
    "Vegetables":      ["Onion_Price", "Tomato_Price", "Potato_Price"],
    "Pulses & Grains": ["Moong", "Tur", "Gram", "Soya"],
    "Oils & Fuel":     ["Mustard", "Palm", "Fuel_Price", "Gur"],
    "Staples":         ["Milk", "Sugar", "Salt", "Tea"],
}

# ── COVID / Event Annotations for Top-10 Chart ───────────────────
EVENT_LABELS = {
    "2020-02": "Pre-COVID\nDisruption",
    "2020-03": "COVID\nLockdown 1",
    "2020-04": "COVID\nLockdown 1",
    "2020-05": "Lockdown\nEase",
    "2020-09": "Unlock\nPhase",
    "2021-01": "2nd Wave\nFear",
    "2021-04": "COVID\nLockdown 2",
    "2021-05": "COVID\nLockdown 2",
    "2021-06": "Post-Lockdown\nSupply Crunch",
    "2021-09": "Post-Monsoon\nSupply Shock",
    "2021-10": "Post-Monsoon\nSupply Shock",
    "2021-11": "Diwali\nDemand Spike",
    "2022-01": "Global Commodity\nPrice Surge",
    "2022-03": "Ukraine War\nOil Shock",
    "2022-10": "Diwali\nDemand Spike",
}

# ── Stacked bar commodity groups for Top-10 chart ────────────────
STACKED_GROUPS = {
    "Vegetables":  ["Onion_Price","Tomato_Price","Potato_Price"],
    "Pulses":      ["Moong","Tur","Gram","Soya"],
    "Oils":        ["Mustard","Palm","Gur"],
    "Fuel":        ["Fuel_Price"],
    "Staples":     ["Milk","Sugar","Salt","Tea"],
}
STACKED_COLORS = {
    "Vegetables": "#e74c3c",
    "Pulses":     "#f39c12",
    "Oils":       "#8e44ad",
    "Fuel":       "#c0392b",
    "Staples":    "#27ae60",
}

DASHBOARD_COLORS = {
    "Vegetables":      ["#e74c3c","#c0392b","#e67e22","#d35400"],
    "Pulses & Grains": ["#f39c12","#d68910","#8e44ad","#6c3483"],
    "Oils & Fuel":     ["#27ae60","#1e8449","#0891B2","#0e6655"],
    "Staples":         ["#2980b9","#1a5276","#16a085","#0e6655"],
}

def showfig(title=""):
    if title:
        plt.suptitle(title, fontsize=13, fontweight="bold") if not plt.gcf().get_suptitle() else None
    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
FESTIVAL_MONTHS = {
    (2019,10):"Diwali",  (2020,10):"Diwali",  (2021,11):"Diwali",
    (2022,10):"Diwali",  (2023,11):"Diwali",
    (2020, 3):"Lockdown1",(2020, 4):"Lockdown1",(2020, 5):"Lockdown1",
    (2021, 4):"Lockdown2",(2021, 5):"Lockdown2",
    (2020, 1):"COVID_Start",
    (2019, 8):"Eid",     (2020, 5):"Eid",      (2021, 5):"Eid",
    (2022, 5):"Eid",     (2023, 4):"Eid",
}

def get_season(month):
    if month in [12,1,2]:    return "Winter"
    elif month in [3,4,5]:   return "Summer"
    elif month in [6,7,8,9]: return "Monsoon"
    else:                    return "PostMonsoon"

def add_features(df, price_col, lags=[1,2,3,6,12]):
    d = df.copy().sort_values("Date").reset_index(drop=True)
    d["Month"]       = d["Date"].dt.month
    d["Month_Name"]  = d["Month"].map(lambda m: MONTH_LABELS[m-1])
    d["Year"]        = d["Date"].dt.year
    d["Season"]      = d["Month"].map(get_season)
    d["Season_Code"] = d["Season"].map({"Winter":0,"Summer":1,"Monsoon":2,"PostMonsoon":3})
    d["Festival"]    = d.apply(lambda r: 1 if (r["Year"],r["Month"]) in FESTIVAL_MONTHS else 0, axis=1)
    d["Festival_Label"] = d.apply(lambda r: FESTIVAL_MONTHS.get((r["Year"],r["Month"]),""), axis=1)
    for lag in lags:
        d[f"Lag_{lag}"] = d[price_col].shift(lag)
    d["Roll_Mean_3"] = d[price_col].rolling(3, min_periods=1).mean()
    d["Roll_Std_3"]  = d[price_col].rolling(3, min_periods=1).std().fillna(0)
    d["Roll_Mean_6"] = d[price_col].rolling(6, min_periods=1).mean()
    d["Roll_Std_6"]  = d[price_col].rolling(6, min_periods=1).std().fillna(0)
    d["MoM_Pct"]    = d[price_col].pct_change() * 100
    d["Month_Sin"]  = np.sin(2*np.pi*d["Month"]/12)
    d["Month_Cos"]  = np.cos(2*np.pi*d["Month"]/12)
    return d

# ══════════════════════════════════════════════════════════════════
# SCALER
# ══════════════════════════════════════════════════════════════════
def get_scaler(scaler_type="minmax"):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    return StandardScaler() if scaler_type == "standard" else MinMaxScaler(feature_range=(0,1))

# ══════════════════════════════════════════════════════════════════
# TIME-SERIES CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════
def tscv_splits(n, n_folds=3, min_train=12):
    fold_size = max(FORECAST_MONTHS, (n - min_train) // n_folds)
    splits = []
    for fold in range(n_folds):
        val_end   = n - fold * fold_size
        val_start = val_end - fold_size
        if val_start < min_train: break
        splits.append((list(range(val_start)), list(range(val_start, val_end))))
    return list(reversed(splits))

def cross_validate_sarima(series, n_folds=3):
    series = np.array(series, dtype=float)
    min_train = max(12, int(len(series)*0.4))
    splits = tscv_splits(len(series), n_folds, min_train=min_train)
    if not splits:
        print(f"    TSCV skipped — not enough data (n={len(series)})")
        return None
    fold_mapes = []
    print(f"    TSCV SARIMA ({len(splits)} folds)...")
    for fi, (tr, va) in enumerate(splits):
        try:
            fm,_,_,_ = forecast_sarima(series[tr], len(va))
            m = calculate_metrics(series[va], fm)
            if m["MAPE"] < 500:
                fold_mapes.append(m["MAPE"])
                print(f"      Fold {fi+1}: train={len(tr)}  val={len(va)}  MAPE={m['MAPE']:.2f}%")
            else:
                print(f"      Fold {fi+1}: MAPE={m['MAPE']:.1f}% (unreliable — excluded)")
        except Exception as e:
            print(f"      Fold {fi+1}: FAILED ({e})")
    if not fold_mapes:
        print("    CV: no valid folds"); return None
    mean_mape = np.mean(fold_mapes)
    print(f"    CV Mean MAPE ({len(fold_mapes)} valid folds): {mean_mape:.2f}%")
    return mean_mape

# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def to_monthly(df, col, agg="mean"):
    d = df.copy()
    d["YM"] = d["Date"].dt.to_period("M")
    m = d.groupby("YM")[col].agg(agg).reset_index()
    m["Date"] = m["YM"].dt.to_timestamp()
    return m.sort_values("Date").reset_index(drop=True)

def detect_anomalies(s, threshold=2.0):
    z = (s - s.mean()) / s.std()
    return z > threshold, z < -threshold

def rolling_avg(s, w):
    return s.rolling(w, min_periods=1).mean()

def fmt_ax(ax, interval=3):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)

def calculate_metrics(actual, predicted, label=""):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    n = min(len(actual), len(predicted))
    actual, predicted = actual[:n], predicted[:n]
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]
    if len(actual) == 0:
        return {"MAE":999, "RMSE":999, "MAPE":999, "SMAPE":999}
    MAE   = mean_absolute_error(actual, predicted)
    RMSE  = np.sqrt(mean_squared_error(actual, predicted))
    denom = np.where(actual == 0, 1, actual)
    MAPE  = min(np.mean(np.abs((actual - predicted) / denom)) * 100, 999.0)
    SMAPE = min(np.mean(2*np.abs(predicted-actual)/(np.abs(actual)+np.abs(predicted)+1e-8))*100, 999.0)
    if label:
        print(f"    [{label}]  MAE={MAE:.2f}  RMSE={RMSE:.2f}  MAPE={MAPE:.2f}%  SMAPE={SMAPE:.2f}%")
    return {"MAE":round(MAE,2), "RMSE":round(RMSE,2), "MAPE":round(MAPE,2), "SMAPE":round(SMAPE,2)}

def get_d(series):
    from statsmodels.tsa.stattools import adfuller
    series = pd.Series(series).dropna()
    if len(series) < 4: return 1
    try:
        _, p, *_ = adfuller(series, autolag="AIC")
        return 0 if p < 0.05 else 1
    except: return 1

def safe_periodicity(series_len):
    for s in [12, 6, 4, 3, 2]:
        if series_len > 2*s + 4: return s
    return 0

def fit_sarima_safe(series, order, seasonal_order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _, _, _, s_seas = seasonal_order
    if s_seas <= 1: seasonal_order = (0,0,0,0)
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    for method in ["lbfgs","powell","nm","bfgs"]:
        try: return model.fit(disp=False, maxiter=200, method=method)
        except: continue
    return None

def forecast_sarima(series, steps):
    series = pd.Series(series).dropna().reset_index(drop=True)
    if len(series) < 4: raise RuntimeError(f"Series too short (n={len(series)})")
    d = get_d(series); S = safe_periodicity(len(series))
    fit = fit_sarima_safe(series, (1,d,1), (1,1,1,S)) if S >= 2 else None
    if fit is None and S >= 2: fit = fit_sarima_safe(series, (1,d,0), (0,1,0,S))
    if fit is None: fit = fit_sarima_safe(series, (1,d,1), (0,0,0,0))
    if fit is None: fit = fit_sarima_safe(series, (1,d,0), (0,0,0,0))
    if fit is None: raise RuntimeError("SARIMA failed all optimizers")
    fc = fit.get_forecast(steps=steps); ci = fc.conf_int(alpha=0.20)
    return fc.predicted_mean, ci.iloc[:,0], ci.iloc[:,1], fit.aic

def gridsearch_sarima(series, steps):
    series = pd.Series(series).dropna().reset_index(drop=True)
    if len(series) < 4: raise RuntimeError(f"Too short for GridSearch (n={len(series)})")
    print("    Running GridSearch SARIMA...")
    d = get_d(series); S = safe_periodicity(len(series))
    best_aic, best_fit, best_order = np.inf, None, None
    for idx, (p,q,P,Q) in enumerate(itertools.product([0,1,2],[0,1,2],[0,1],[0,1])):
        if S < 2 and (P > 0 or Q > 0): continue
        s_ord = (P,1,Q,S) if S >= 2 else (0,0,0,0)
        try:
            fit = fit_sarima_safe(series, (p,d,q), s_ord)
            if fit is None: continue
            if fit.aic < best_aic: best_aic, best_fit, best_order = fit.aic, fit, ((p,d,q), s_ord)
            print(f"      [{idx+1}] ({p},{d},{q})x{s_ord}  AIC={fit.aic:.1f}")
        except: continue
    if best_fit is None: raise RuntimeError("GridSearch: no model converged")
    print(f"    Best: {best_order}  AIC={best_aic:.1f}")
    fc = best_fit.get_forecast(steps=steps); ci = fc.conf_int(alpha=0.20)
    return fc.predicted_mean, ci.iloc[:,0], ci.iloc[:,1], best_aic, best_order

def randomsearch_sarima(series, steps, n_iter=15):
    series = pd.Series(series).dropna().reset_index(drop=True)
    if len(series) < 4: raise RuntimeError(f"Too short for RandomSearch (n={len(series)})")
    print(f"    Running RandomSearch SARIMA ({n_iter} iters)...")
    d = get_d(series); S = safe_periodicity(len(series))
    best_aic, best_fit, best_order = np.inf, None, None
    success = 0; random.seed(42)
    for _ in range(n_iter*3):
        if success >= n_iter: break
        p = random.choice([0,1,2]); q = random.choice([0,1,2])
        P = random.choice([0,1]) if S >= 2 else 0
        Q = random.choice([0,1]) if S >= 2 else 0
        s_ord = (P,1,Q,S) if S >= 2 else (0,0,0,0)
        try:
            fit = fit_sarima_safe(series, (p,d,q), s_ord)
            if fit is None: continue
            if fit.aic < best_aic: best_aic, best_fit, best_order = fit.aic, fit, ((p,d,q), s_ord)
            print(f"      [{success+1}/{n_iter}] ({p},{d},{q})x{s_ord}  AIC={fit.aic:.1f}")
            success += 1
        except: continue
    if best_fit is None: raise RuntimeError("RandomSearch: no model converged")
    print(f"    Best: {best_order}  AIC={best_aic:.1f}")
    fc = best_fit.get_forecast(steps=steps); ci = fc.conf_int(alpha=0.20)
    return fc.predicted_mean, ci.iloc[:,0], ci.iloc[:,1], best_aic, best_order

def forecast_prophet(dates, values, steps, changepoint_prior=0.3, series_name=""):
    from prophet import Prophet
    volatile_series = ["Onion_Price","Tomato_Price","Potato_Price","Veg_Price"]
    stable_series   = ["Sugar","Salt","Milk","Gram","Tur","Moong"]
    if series_name in volatile_series:
        cp_scale  = 0.5
        seas_scale = 20
    elif series_name in stable_series:
        cp_scale  = 0.05
        seas_scale = 5
    else:
        cp_scale  = changepoint_prior
        seas_scale = 15
    df_p = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values}).dropna()
    if len(df_p) < 2: raise RuntimeError(f"Prophet needs ≥2 rows, got {len(df_p)}")
    m = Prophet(
        changepoint_prior_scale  = cp_scale,
        seasonality_prior_scale  = seas_scale,
        yearly_seasonality       = len(df_p) >= 24,
        weekly_seasonality       = False,
        daily_seasonality        = False,
        interval_width           = 0.80,
        n_changepoints           = min(25, len(df_p) // 3),
    )
    m.fit(df_p)
    future = m.make_future_dataframe(periods=steps, freq="MS")
    fc = m.predict(future)
    fut = fc[fc["ds"] > df_p["ds"].max()]
    return fut["yhat"].values, fut["yhat_lower"].values, fut["yhat_upper"].values

def forecast_lstm(series, steps, look_back=12, epochs=80, units=50):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel("ERROR")
    except Exception:
        print("    [LSTM] TensorFlow not available — skipping"); return None, None, None
    print(f"    Training LSTM (look_back={look_back}, scaler={SCALER_TYPE})...")
    series = np.array(series, dtype=float)
    if len(series) < 6:
        print(f"    [LSTM] Too short (n={len(series)}) — skipping"); return None, None, None
    if np.any(np.isnan(series)): series = pd.Series(series).ffill().bfill().values
    scaler = get_scaler(SCALER_TYPE)
    scaled = scaler.fit_transform(series.reshape(-1,1))
    look_back = max(1, look_back)
    if len(scaled) < look_back + steps: look_back = max(1, len(scaled) - steps - 1)
    if look_back < 1: return None, None, None
    X, y_seq = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0]); y_seq.append(scaled[i, 0])
    if len(X) < 5: return None, None, None
    X = np.array(X).reshape(-1, look_back, 1); y_seq = np.array(y_seq)
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(look_back,1)),
        Dropout(0.2), LSTM(units//2), Dropout(0.2),
        Dense(16, activation="relu"), Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber")
    val_split = 0.15 if len(X) > 20 else 0.0
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
    model.fit(X, y_seq, epochs=epochs, batch_size=min(16, max(1, len(X)//2)),
              validation_split=val_split,
              callbacks=[early_stop] if val_split > 0 else [], verbose=0)
    last_seq = scaled[-look_back:].flatten(); preds = []
    for _ in range(steps):
        inp = last_seq.reshape(1, look_back, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        if SCALER_TYPE == "minmax": pred = np.clip(pred, 0, 1)
        preds.append(pred); last_seq = np.append(last_seq[1:], pred)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    print("    LSTM complete")
    return preds, preds*0.92, preds*1.08

def forecast_hybrid_sarima_lstm(series, steps, look_back=6, epochs=60):
    print("    Training Hybrid SARIMA+LSTM...")
    series = np.array(series, dtype=float)
    if len(series) < 20:
        print("    [Hybrid] Too short — skipping"); return None, None, None
    try:
        sarima_mean, sarima_lo, sarima_hi, _ = forecast_sarima(series, steps)
        sarima_mean = np.array(sarima_mean)
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        d  = get_d(series); S = safe_periodicity(len(series))
        fit = fit_sarima_safe(pd.Series(series), (1,d,1),
                              (1,1,1,S) if S >= 2 else (0,0,0,0))
        if fit is None:
            print("    [Hybrid] SARIMA fit failed — returning SARIMA only")
            return sarima_mean, sarima_lo, sarima_hi
        fitted_vals = fit.fittedvalues.values
        min_len = min(len(series), len(fitted_vals))
        residuals = series[-min_len:] - fitted_vals[-min_len:]
        lb = max(3, min(look_back, len(residuals) // 4))
        res_preds, _, _ = forecast_lstm(residuals, steps, look_back=lb, epochs=epochs, units=32)
        if res_preds is None:
            print("    [Hybrid] LSTM residual model failed — returning SARIMA only")
            return sarima_mean, sarima_lo, sarima_hi
        hybrid_preds = sarima_mean + res_preds[:steps]
        hybrid_lo    = hybrid_preds * 0.91
        hybrid_hi    = hybrid_preds * 1.09
        print("    Hybrid SARIMA+LSTM complete")
        return hybrid_preds, hybrid_lo, hybrid_hi
    except Exception as e:
        print(f"    [Hybrid] FAILED: {e}")
        return None, None, None

def _build_ml_features(arr, lag_max=6):
    rows = []
    for i in range(lag_max, len(arr)):
        row = {f"lag_{l}": arr[i-l] for l in range(1, lag_max+1)}
        row["roll_mean_3"] = np.mean(arr[max(0,i-3):i])
        row["roll_std_3"]  = np.std(arr[max(0,i-3):i])  if i > 3  else 0.0
        row["roll_mean_6"] = np.mean(arr[max(0,i-6):i])
        row["roll_std_6"]  = np.std(arr[max(0,i-6):i])  if i > 6  else 0.0
        m_ = (i % 12) + 1
        row["month_sin"] = np.sin(2*np.pi*m_/12)
        row["month_cos"] = np.cos(2*np.pi*m_/12)
        row["trend"] = i
        rows.append(row)
    return pd.DataFrame(rows)

def _predict_ml_steps(model, scaler, series, steps, feature_names, lag_max=6):
    extended = list(series.copy()); preds = []
    for _ in range(steps):
        i = len(extended)
        row = {f"lag_{l}": extended[i-l] if i-l >= 0 else 0.0 for l in range(1, lag_max+1)}
        row["roll_mean_3"] = np.mean(extended[max(0,i-3):i])
        row["roll_std_3"]  = np.std(extended[max(0,i-3):i])  if i > 3  else 0.0
        row["roll_mean_6"] = np.mean(extended[max(0,i-6):i])
        row["roll_std_6"]  = np.std(extended[max(0,i-6):i])  if i > 6  else 0.0
        m_ = (i % 12) + 1
        row["month_sin"] = np.sin(2*np.pi*m_/12)
        row["month_cos"] = np.cos(2*np.pi*m_/12)
        row["trend"] = i
        X_new = scaler.transform(pd.DataFrame([row])[feature_names].values)
        p_ = model.predict(X_new)[0]; preds.append(p_); extended.append(p_)
    return np.array(preds)

def forecast_xgboost(series, steps, series_name="Series", lag_max=6,
                     n_estimators=300, max_depth=4, learning_rate=0.05):
    try:
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("    [XGBoost] Not installed — skipping"); return None, None, None
    print("    Training XGBoost...")
    series = np.array(series, dtype=float)
    if np.any(np.isnan(series)): series = pd.Series(series).ffill().bfill().values
    X_df = _build_ml_features(series, lag_max); y_arr = series[lag_max:]
    if len(X_df) < 10: return None, None, None
    feature_names = X_df.columns.tolist()
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_df.values)
    model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate, subsample=0.8,
                             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                             random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_scaled, y_arr)
    preds = _predict_ml_steps(model, scaler, series, steps, feature_names, lag_max)
    print("    XGBoost complete")
    return preds, preds*0.92, preds*1.08

def forecast_random_forest(series, steps, series_name="Series", lag_max=6,
                           n_estimators=300, max_depth=8):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("    [RF] scikit-learn not installed — skipping"); return None, None, None
    print("    Training Random Forest...")
    series = np.array(series, dtype=float)
    if np.any(np.isnan(series)): series = pd.Series(series).ffill().bfill().values
    X_df = _build_ml_features(series, lag_max); y_arr = series[lag_max:]
    if len(X_df) < 10: return None, None, None
    feature_names = X_df.columns.tolist()
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_df.values)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  random_state=42, n_jobs=-1)
    model.fit(X_scaled, y_arr)
    preds = _predict_ml_steps(model, scaler, series, steps, feature_names, lag_max)
    print("    Random Forest complete")
    return preds, preds*0.92, preds*1.08

def rolling_forecast_sarima(series, dates, test_size=6):
    print("    Running Rolling Walk-Forward Forecast...")
    if len(series) < test_size + 4:
        raise RuntimeError(f"Too short for rolling forecast (n={len(series)})")
    d = get_d(series); S = safe_periodicity(len(series) - test_size)
    preds, actuals = [], []
    train_end = len(series) - test_size
    for i in range(test_size):
        train = series[:train_end+i]; actual_val = series[train_end+i]
        fit = fit_sarima_safe(pd.Series(train).reset_index(drop=True), (1,d,1),
                              (1,1,1,S) if S >= 2 else (0,0,0,0))
        pred = fit.forecast(steps=1).iloc[0] if fit else np.mean(train[-3:])
        preds.append(pred); actuals.append(actual_val)
        print(f"      Step {i+1}/{test_size}: actual={actual_val:.2f}  pred={pred:.2f}  err={abs(actual_val-pred):.2f}")
    print("    Rolling Forecast complete")
    return np.array(preds), np.array(actuals), dates.iloc[train_end:train_end+test_size]

def remove_outliers(series, method="iqr"):
    s = series.copy().astype(float)
    if method == "iqr":
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        mask = (s < lo) | (s > hi)
    else:
        z = (s - s.mean()) / s.std()
        mask = z.abs() > 3
    s[mask] = np.nan
    s = s.interpolate(method="linear").ffill().bfill()
    return s, mask

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("="*65); print("  LOADING DATA"); print("="*65)

fuel_raw = pd.read_csv("datasets/fuel_by_date.csv")
fuel_raw.columns = [c.strip() for c in fuel_raw.columns]
for old, new in [("date","Date"),("rate","Fuel_Price"),("Rate","Fuel_Price"),
                 ("fuel_price","Fuel_Price"),("price","Fuel_Price")]:
    if old in fuel_raw.columns and new not in fuel_raw.columns:
        fuel_raw.rename(columns={old:new}, inplace=True)
fuel_raw["Date"] = pd.to_datetime(fuel_raw["Date"], dayfirst=True, errors="coerce")
fuel_raw.dropna(subset=["Date"], inplace=True)
fuel_raw = fuel_raw.sort_values("Date").reset_index(drop=True)

veg_raw = pd.read_csv("datasets/vegetable_inflation_dataset.csv")
veg_raw.columns = [c.strip() for c in veg_raw.columns]
veg_raw = veg_raw.loc[:, ~veg_raw.columns.str.contains('^Unnamed')]
print(f"  veg_raw columns: {veg_raw.columns.tolist()}")

veg_raw["Date"] = pd.to_datetime(veg_raw["Date"], format="%Y-%m-%d", errors="coerce")
veg_raw.dropna(subset=["Date"], inplace=True)
veg_raw = veg_raw.sort_values("Date").reset_index(drop=True)

if "Average_Price" in veg_raw.columns:
    veg_raw.rename(columns={"Average_Price": "Veg_Price"}, inplace=True)
elif "Average" in veg_raw.columns:
    veg_raw.rename(columns={"Average": "Veg_Price"}, inplace=True)
else:
    num_cols = veg_raw.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        veg_raw.rename(columns={num_cols[0]: "Veg_Price"}, inplace=True)
    else:
        raise ValueError("No numeric column found in vegetable_inflation_dataset.csv")

veg_m = veg_raw[["Date","Veg_Price"]].copy()
veg_m["Date"] = veg_m["Date"].dt.to_period("M").dt.to_timestamp()
if veg_m.duplicated("Date").any():
    veg_m = veg_m.groupby("Date")["Veg_Price"].mean().reset_index()
veg_m = veg_m.sort_values("Date").reset_index(drop=True)
print(f"  ✓ veg_m rows: {len(veg_m)}  |  {veg_m['Date'].min().date()} → {veg_m['Date'].max().date()}")

grocery_raw = pd.read_csv("datasets/merged_grocery_dataset.csv")
grocery_raw.columns = [c.strip() for c in grocery_raw.columns]
grocery_raw["Date"] = pd.to_datetime(grocery_raw["Date"], dayfirst=True, errors="coerce")
grocery_raw.dropna(subset=["Date"], inplace=True)
grocery_raw = grocery_raw.sort_values("Date").reset_index(drop=True)

GROCERY_COLS = ["Tea","Salt","Mustard","Palm","Wheat",
                "Milk","Sugar","Gur","Gram","Moong","Soya","Tur"]
commodity_dfs = {}
for col_name in GROCERY_COLS:
    if col_name in grocery_raw.columns:
        df_tmp = grocery_raw[["Date",col_name]].dropna(subset=[col_name]).copy()
        df_tmp["YM"] = df_tmp["Date"].dt.to_period("M")
        m_tmp = df_tmp.groupby("YM")[col_name].mean().reset_index()
        m_tmp["Date"] = m_tmp["YM"].dt.to_timestamp()
        m_tmp = m_tmp.sort_values("Date").reset_index(drop=True)
        commodity_dfs[col_name] = m_tmp
        print(f"  ✓ {col_name:<16}: {m_tmp['Date'].min().date()} → {m_tmp['Date'].max().date()}  (n={len(m_tmp)})")

VEG_COLS = ["Onion_Price","Potato_Price","Tomato_Price"]
for col_name in VEG_COLS:
    if col_name in veg_raw.columns:
        df_tmp = veg_raw[["Date",col_name]].dropna(subset=[col_name]).copy()
        df_tmp["YM"] = df_tmp["Date"].dt.to_period("M")
        m_tmp = df_tmp.groupby("YM")[col_name].mean().reset_index()
        m_tmp["Date"] = m_tmp["YM"].dt.to_timestamp()
        m_tmp = m_tmp.sort_values("Date").reset_index(drop=True)
        commodity_dfs[col_name] = m_tmp
        print(f"  ✓ {col_name:<16}: {m_tmp['Date'].min().date()} → {m_tmp['Date'].max().date()}  (n={len(m_tmp)})")

fuel_m = to_monthly(fuel_raw, "Fuel_Price", "mean")
print(f"  ✓ {'Fuel_Price':<16}: {fuel_m['Date'].min().date()} → {fuel_m['Date'].max().date()}  (n={len(fuel_m)})")

# ══════════════════════════════════════════════════════════════════
# BUILD TARGETS + OUTLIER REMOVAL + FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
COLORS = ["#2ecc71","#e67e22","#3498db","#9b59b6","#f39c12","#e74c3c",
          "#1abc9c","#e91e63","#00bcd4","#8bc34a","#ff5722","#607d8b",
          "#795548","#f1c40f","#d63031","#6c5ce7","#0984e3","#fd79a8"]

print(f"\n  Applying outlier removal (method={OUTLIER_METHOD})...")
for df_, col_ in [(veg_m,"Veg_Price"),(fuel_m,"Fuel_Price")]:
    cleaned, _ = remove_outliers(df_[col_], method=OUTLIER_METHOD); df_[col_] = cleaned
for col_name, df_ in commodity_dfs.items():
    cleaned, _ = remove_outliers(df_[col_name], method=OUTLIER_METHOD); df_[col_name] = cleaned

forecast_targets = [
    {"label":"Veg_Price",  "series":veg_m,  "col":"Veg_Price",  "color":COLORS[0]},
    {"label":"Fuel_Price", "series":fuel_m, "col":"Fuel_Price", "color":COLORS[1]},
]
for i, (col_name, df) in enumerate(commodity_dfs.items()):
    forecast_targets.append({"label":col_name, "series":df, "col":col_name, "color":COLORS[i+2]})

print("  Building enriched feature tables...")
for tgt in forecast_targets:
    tgt["enriched"] = add_features(tgt["series"], tgt["col"])
    print(f"    {tgt['col']}: {len(tgt['series'])} months  festivals={tgt['enriched']['Festival'].sum()}")

n = len(forecast_targets)
print(f"\n  Total series: {n}")

import seaborn as sns

tgt_lookup = {t["col"]: t for t in forecast_targets}

# ══════════════════════════════════════════════════════════════════
# NOTE: Outlier visualisation removed as requested
# ══════════════════════════════════════════════════════════════════

# ── 00c Season & festival analysis ───────────────────────────────
print("00c Season & festival analysis...")
for tgt in forecast_targets[:4]:
    col = tgt["col"]; enr = tgt["enriched"]
    fig_sf, axes_sf = plt.subplots(1, 2, figsize=(14,5))
    fig_sf.suptitle(f"{col} — Season & Festival Analysis", fontsize=12, fontweight="bold")
    ax_s = axes_sf[0]
    season_order = ["Winter","Summer","Monsoon","PostMonsoon"]
    data_by_season = [enr[enr["Season"]==s][col].dropna().values for s in season_order]
    bp = ax_s.boxplot(data_by_season, labels=season_order, patch_artist=True,
                      medianprops={"color":"black","lw":2})
    for patch, sc in zip(bp["boxes"], ["#74b9ff","#fd79a8","#00b894","#fdcb6e"]):
        patch.set_facecolor(sc)
    ax_s.set_title("Price Distribution by Season"); ax_s.set_ylabel("₹")
    ax_s.grid(axis="y", linestyle="--", alpha=0.3)
    ax_f = axes_sf[1]
    fest_vals = enr[enr["Festival"]==1][col].dropna()
    nonfest_vals = enr[enr["Festival"]==0][col].dropna()
    bp2 = ax_f.boxplot([nonfest_vals, fest_vals], labels=["Normal","Festival"],
                       patch_artist=True, medianprops={"color":"black","lw":2})
    bp2["boxes"][0].set_facecolor("#dfe6e9"); bp2["boxes"][1].set_facecolor("#fab1a0")
    ax_f.set_title("Festival vs Normal Price"); ax_f.set_ylabel("₹")
    ax_f.grid(axis="y", linestyle="--", alpha=0.3)
    if len(fest_vals) > 0 and len(nonfest_vals) > 0:
        diff = fest_vals.mean() - nonfest_vals.mean()
        ax_f.text(1.5, fest_vals.max(), f"Δ = ₹{diff:.1f}", ha="center",
                  fontsize=9, color="#d63031", fontweight="bold")
    plt.tight_layout(); plt.show()

# ══════════════════════════════════════════════════════════════════
# THEMED DASHBOARDS  — Time Series
# ══════════════════════════════════════════════════════════════════
print("\n01 Themed Time-Series Dashboards (4 separate figures)...")

for dashboard_name, group_cols in DASHBOARD_GROUPS.items():
    available = [c for c in group_cols if c in tgt_lookup]
    if not available:
        print(f"  Skipping {dashboard_name} — no data available"); continue

    ncols = 2; nrows = (len(available) + 1) // 2
    fig_d, axes_d = plt.subplots(nrows, ncols, figsize=(16, nrows * 5),
                                  facecolor="#f8f9fa")
    fig_d.suptitle(f"Monthly Time Series — {dashboard_name}",
                   fontsize=14, fontweight="bold", y=1.01)
    axes_d = axes_d.flatten() if nrows > 1 else [axes_d] if ncols == 1 else list(axes_d.flatten())

    dash_colors = DASHBOARD_COLORS.get(dashboard_name, PALETTE)

    for idx, (ax, col) in enumerate(zip(axes_d, available)):
        tgt = tgt_lookup[col]
        clr = dash_colors[idx % len(dash_colors)]
        m_  = tgt["series"]
        dates = m_["Date"]; vals = m_[col]
        ma3  = rolling_avg(vals, 3)
        ma12 = rolling_avg(vals, 12)
        spikes, dips = detect_anomalies(vals)

        ax.set_facecolor("#fdfdfd")
        ax.fill_between(dates, vals, alpha=0.1, color=clr)
        ax.bar(dates, vals, width=20, color=clr, alpha=0.18, label="Monthly")
        ax.plot(dates, ma3,  color=clr,     lw=2.0, label="3-MA")
        ax.plot(dates, ma12, color="#7F8C8D", lw=2.2, ls="--", label="12-MA")

        if spikes.sum():
            ax.scatter(dates[spikes], vals[spikes], color="#E74C3C", s=70, zorder=5,
                       label=f"Spike({spikes.sum()})", edgecolors="white", lw=0.8)
        if dips.sum():
            ax.scatter(dates[dips], vals[dips], color="#8E44AD", s=70, zorder=5,
                       label=f"Dip({dips.sum()})", edgecolors="white", lw=0.8)

        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("₹", fontsize=9)
        ax.legend(fontsize=8, loc="upper left", ncol=3)
        fmt_ax(ax, interval=4)

    for ax in axes_d[len(available):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════
# 02 — SEASONAL PATTERNS — Themed Dashboard (matching 01 style)
# ══════════════════════════════════════════════════════════════════
print("02 Themed Seasonal Patterns Dashboards...")
for dashboard_name, group_cols in DASHBOARD_GROUPS.items():
    available = [c for c in group_cols if c in tgt_lookup]
    if not available: continue

    ncols = 2; nrows = (len(available) + 1) // 2
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(16, nrows * 5),
                                facecolor="#f8f9fa")
    fig2.suptitle(f"Seasonal Patterns — {dashboard_name}",
                  fontsize=14, fontweight="bold", y=1.01)
    axes2 = axes2.flatten() if nrows > 1 else list(np.array([axes2]).flatten())
    dash_colors = DASHBOARD_COLORS.get(dashboard_name, PALETTE)

    for idx, (ax, col) in enumerate(zip(axes2, available)):
        tgt = tgt_lookup[col]
        clr = dash_colors[idx % len(dash_colors)]
        m2  = tgt["series"].copy()
        m2["Month"] = m2["Date"].dt.month
        avg = m2.groupby("Month")[col].mean()
        bar_colors = ["#c0392b" if i+1 == avg.idxmax() else clr for i in range(len(avg))]

        ax.set_facecolor("#fdfdfd")
        ax.bar(avg.index, avg.values, color=bar_colors, alpha=0.85, width=0.65,
               edgecolor="white", lw=0.8)
        ax.set_xticks(avg.index)
        ax.set_xticklabels([MONTH_LABELS[i-1] for i in avg.index],
                           fontsize=8, rotation=45, ha="right")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Avg ₹")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        pk = avg.idxmax()
        ax.text(pk, avg[pk]*1.02, f"Peak\n{MONTH_LABELS[pk-1]}",
                fontsize=7, color="#c0392b", ha="center", va="bottom", fontweight="bold")

    for ax in axes2[len(available):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════
# 03 — YEAR-OVER-YEAR — Themed Dashboard (matching 01 style)
# ══════════════════════════════════════════════════════════════════
print("03 Year-over-year (Themed Dashboard)...")
for dashboard_name, group_cols in DASHBOARD_GROUPS.items():
    available = [c for c in group_cols if c in tgt_lookup]
    if not available: continue

    ncols = 2; nrows = (len(available) + 1) // 2
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(16, nrows * 5),
                                facecolor="#f8f9fa")
    fig3.suptitle(f"Year-over-Year Comparison — {dashboard_name}",
                  fontsize=14, fontweight="bold", y=1.01)
    axes3 = axes3.flatten() if nrows > 1 else list(np.array([axes3]).flatten())
    dash_colors = DASHBOARD_COLORS.get(dashboard_name, PALETTE)

    for idx, (ax, col) in enumerate(zip(axes3, available)):
        tgt = tgt_lookup[col]
        clr = dash_colors[idx % len(dash_colors)]
        m_  = tgt["series"].copy()
        m_["year"]  = m_["Date"].dt.year
        m_["month"] = m_["Date"].dt.month
        years = sorted(m_["year"].unique())
        cmap_ = plt.cm.Blues(np.linspace(0.35, 0.95, max(len(years), 1)))

        ax.set_facecolor("#fdfdfd")
        for i, yr in enumerate(years):
            sub = m_[m_["year"]==yr].sort_values("month")
            ax.plot(sub["month"], sub[col], color=cmap_[i], lw=1.8,
                    marker="o", ms=3, label=str(yr))

        ax.set_xticks(range(1,13))
        ax.set_xticklabels(MONTH_LABELS, fontsize=8, rotation=45, ha="right")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("₹", fontsize=9)
        ax.legend(title="Year", fontsize=7, loc="upper left", ncol=2)
        ax.grid(linestyle="--", alpha=0.3)

    for ax in axes3[len(available):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════
# 04 — SEASONAL HEATMAP — Themed Dashboard (matching 01 style)
# ══════════════════════════════════════════════════════════════════
print("04 Seasonal heatmap (Themed Dashboard)...")
for dashboard_name, group_cols in DASHBOARD_GROUPS.items():
    available = [c for c in group_cols if c in tgt_lookup]
    if not available: continue

    ncols = 2; nrows = (len(available) + 1) // 2
    fig4, axes4 = plt.subplots(nrows, ncols, figsize=(16, nrows * 5),
                                facecolor="#f8f9fa")
    fig4.suptitle(f"Seasonal Heatmap (Year × Month) — {dashboard_name}",
                  fontsize=14, fontweight="bold", y=1.01)
    axes4 = axes4.flatten() if nrows > 1 else list(np.array([axes4]).flatten())

    for idx, (ax, col) in enumerate(zip(axes4, available)):
        tgt = tgt_lookup[col]
        m_  = tgt["series"].copy()
        m_["year"]  = m_["Date"].dt.year
        m_["month"] = m_["Date"].dt.month
        pivot = m_.pivot_table(index="year", columns="month", values=col, aggfunc="mean")
        pivot.columns = [MONTH_LABELS[c-1] for c in pivot.columns]

        ax.set_facecolor("#fdfdfd")
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
                    linewidths=0.4, cbar_kws={"shrink":0.7}, annot_kws={"size":7})
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("Year")

    for ax in axes4[len(available):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

# ── 05 Correlation scatter ────────────────────────────────────────
print("05 Correlation scatter...")
all_monthly = {}
for tgt in forecast_targets:
    s = tgt["series"].set_index("Date")[tgt["col"]]
    s.index = s.index.to_period("M"); all_monthly[tgt["col"]] = s
merged_all = pd.DataFrame(all_monthly).dropna()
valid_cols = list(merged_all.columns)
pairs = [(x,y) for x,y in [("Fuel_Price","Veg_Price"),("Fuel_Price","Tea"),("Veg_Price","Sugar"),
                             ("Onion_Price","Tomato_Price"),("Potato_Price","Onion_Price")]
         if x in valid_cols and y in valid_cols]
if not pairs and len(valid_cols) >= 2: pairs = [(valid_cols[0], valid_cols[1])]
if pairs:
    fig5, axes5 = plt.subplots(1, len(pairs), figsize=(6*len(pairs), 5))
    if len(pairs) == 1: axes5 = [axes5]
    fig5.suptitle("Cross-Series Correlation Scatter", fontsize=13, fontweight="bold")
    for ax, (x,y), c in zip(axes5, pairs, PALETTE):
        pair_df = merged_all[[x,y]].dropna()
        ax.scatter(pair_df[x], pair_df[y], alpha=0.6, color=c, s=30, edgecolors="white", lw=0.5)
        if len(pair_df) >= 2:
            m_c, b_c = np.polyfit(pair_df[x], pair_df[y], 1)
            xline = np.linspace(pair_df[x].min(), pair_df[x].max(), 100)
            ax.plot(xline, m_c*xline+b_c, color="black", lw=1.5, ls="--")
        r = pair_df.corr().iloc[0,1]
        ax.set_title(f"{x} vs {y}\nr = {r:.3f}"); ax.set_xlabel(x); ax.set_ylabel(y)
        ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.show()

# ── 05b Correlation matrix ────────────────────────────────────────
print("05b Correlation matrix...")
fig5b, ax5b = plt.subplots(figsize=(14,12))
corr_mat = merged_all.corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, ax=ax5b, mask=mask, cmap="RdYlGn", center=0,
            annot=True, fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink":0.8})
ax5b.set_title("Correlation Matrix — All Price Series", pad=15, fontweight="bold")
plt.tight_layout(); plt.show()

# ══════════════════════════════════════════════════════════════════
# 05c — LAG CORRELATION ON FIRST DIFFERENCES (not raw price)
# ══════════════════════════════════════════════════════════════════
print("\n05c Lag Correlation Analysis on First Differences (Δ Price)...")

# Build differenced series — removes non-stationarity, isolates change signals
diff_all = merged_all.diff().dropna()

strong_pairs = []
cols_list = list(diff_all.columns)
for i in range(len(cols_list)):
    for j in range(i+1, len(cols_list)):
        r = diff_all[cols_list[i]].corr(diff_all[cols_list[j]])
        if abs(r) > 0.35:   # lower threshold since diffs are noisier
            strong_pairs.append((cols_list[i], cols_list[j], round(r, 3)))

# Always include key pairs of interest
known_pairs = [("Fuel_Price","Tea"), ("Fuel_Price","Salt"), ("Fuel_Price","Gram"),
               ("Onion_Price","Tomato_Price"), ("Fuel_Price","Veg_Price")]
for kp in known_pairs:
    if kp[0] in diff_all.columns and kp[1] in diff_all.columns:
        if not any(p[0]==kp[0] and p[1]==kp[1] for p in strong_pairs):
            r = diff_all[kp[0]].corr(diff_all[kp[1]])
            strong_pairs.append((kp[0], kp[1], round(r, 3)))

strong_pairs = sorted(strong_pairs, key=lambda x: abs(x[2]), reverse=True)[:6]
print(f"  Strong pairs (on Δ): {[(p[0], p[1], p[2]) for p in strong_pairs]}")

if strong_pairs:
    MAX_LAG = 12
    ncols = min(3, len(strong_pairs))
    nrows = (len(strong_pairs) + ncols - 1) // ncols
    fig_lag, axes_lag = plt.subplots(nrows, ncols,
                                      figsize=(7*ncols, 5*nrows),
                                      facecolor="#f8f9fa")
    fig_lag.suptitle(
        "Lag Correlation — Month-over-Month Δ Price (First Difference)\n"
        "Positive lag = X leads Y  |  Negative lag = Y leads X\n"
        "Using Δ Price removes price-level trends for cleaner signal",
        fontsize=12, fontweight="bold", y=1.02)
    axes_lag = np.array(axes_lag).flatten()

    for ax, (col_x, col_y, static_r) in zip(axes_lag, strong_pairs):
        lags_range = range(-MAX_LAG, MAX_LAG+1)
        lag_corrs  = []
        s_x = diff_all[col_x].dropna()
        s_y = diff_all[col_y].dropna()
        common_idx = s_x.index.intersection(s_y.index)
        s_x = s_x[common_idx]; s_y = s_y[common_idx]

        for lag in lags_range:
            if lag > 0:
                r = s_x.iloc[:-lag].corr(s_y.iloc[lag:])
            elif lag < 0:
                r = s_x.iloc[-lag:].corr(s_y.iloc[:lag])
            else:
                r = s_x.corr(s_y)
            lag_corrs.append(r if not np.isnan(r) else 0)

        lag_arr  = np.array(list(lags_range))
        corr_arr = np.array(lag_corrs)
        best_lag = lag_arr[np.argmax(np.abs(corr_arr))]
        best_r   = corr_arr[np.argmax(np.abs(corr_arr))]

        bar_colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in corr_arr]
        ax.set_facecolor("#fdfdfd")
        ax.bar(lag_arr, corr_arr, color=bar_colors, alpha=0.75, width=0.7, edgecolor="white")
        ax.axhline(0,    color="gray",    lw=0.8, ls="-")
        ax.axhline( 0.35, color="#27ae60", lw=1.0, ls="--", alpha=0.6, label="r=±0.35")
        ax.axhline(-0.35, color="#27ae60", lw=1.0, ls="--", alpha=0.6)
        ax.axvline(best_lag, color="#e74c3c", lw=2.0, ls=":", alpha=0.9)

        ax.set_title(
            f"Δ{col_x}  →  Δ{col_y}\nStatic r={static_r:.2f}  |  Best lag={best_lag}m  r={best_r:.2f}",
            fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag (months)", fontsize=9)
        ax.set_ylabel("Pearson r  (on Δ)", fontsize=9)
        ax.set_xlim(-MAX_LAG-0.5, MAX_LAG+0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if best_lag > 0:
            lead_txt = f"Δ{col_x} leads Δ{col_y}\nby {best_lag} months"
        elif best_lag < 0:
            lead_txt = f"Δ{col_y} leads Δ{col_x}\nby {abs(best_lag)} months"
        else:
            lead_txt = "Simultaneous\nmovement"
        ax.text(0.02, 0.97, lead_txt, transform=ax.transAxes,
                fontsize=8, va="top", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.8))
        ax.legend(fontsize=7)

    for ax in axes_lag[len(strong_pairs):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

# ── 06 Rolling correlation ────────────────────────────────────────
print("06 Rolling correlation...")
fig6, ax6 = plt.subplots(figsize=(13,5))
window = 6; base_col = "Fuel_Price"
if base_col in merged_all.columns:
    others = [c for c in merged_all.columns if c != base_col][:5]
    for other, c_ in zip(others, PALETTE):
        rc = merged_all[base_col].rolling(window).corr(merged_all[other])
        ax6.plot(rc.index.to_timestamp(), rc.values, label=f"Fuel↔{other}", lw=2, color=c_)
ax6.axhline(0, color="gray", lw=0.8, ls="--")
ax6.set_title(f"Rolling {window}-Month Correlation with Fuel Price")
ax6.set_ylabel("Pearson r"); ax6.set_ylim(-1.1, 1.1)
ax6.legend(fontsize=8, ncol=2); ax6.grid(linestyle="--", alpha=0.3)
plt.tight_layout(); plt.show()

# ── 07 Inflation index ────────────────────────────────────────────
print("07 Inflation index...")
fig7, ax7 = plt.subplots(figsize=(13,5))
for tgt, c_ in zip(forecast_targets[:6], PALETTE):
    col = tgt["col"]; s_ = tgt["series"].set_index("Date")[col].dropna()
    base = s_.iloc[0]
    if base != 0: ax7.plot(s_.index, (s_/base)*100, label=col, lw=1.8, color=c_)
ax7.axhline(100, color="gray", lw=0.9, ls="--", label="Base=100")
ax7.set_title("Cumulative Inflation Index (Base = 100)"); ax7.set_ylabel("Index Value")
ax7.legend(fontsize=8, ncol=2); ax7.grid(linestyle="--", alpha=0.3)
plt.tight_layout(); plt.show()

# ── 07b Top-10 Critical Months ────────────────────────────────────
print("07b Top-10 Critical Months — Stacked Bar with Event Annotations...")

changes = {}
for tgt in forecast_targets:
    col = tgt["col"]; s_ = tgt["series"].set_index("Date")[col]
    s_.index = s_.index.to_period("M"); changes[col] = s_.pct_change() * 100
chg_df = pd.DataFrame(changes).dropna()
chg_df["Composite"] = chg_df.mean(axis=1)
top10_idx = chg_df["Composite"].abs().nlargest(10).index
top10_df  = chg_df.loc[top10_idx].sort_values("Composite", ascending=False)

fig7b, ax7b = plt.subplots(figsize=(14, 8), facecolor="#f8f9fa")
ax7b.set_facecolor("#fdfdfd")

x_labels = [i.strftime("%b %Y") for i in top10_df.index.to_timestamp()]
x_pos    = np.arange(len(x_labels))
bottom   = np.zeros(len(x_labels))

for grp_name, grp_cols in STACKED_GROUPS.items():
    available_grp = [c for c in grp_cols if c in top10_df.columns]
    if not available_grp: continue
    grp_vals = top10_df[available_grp].mean(axis=1).values
    ax7b.bar(x_pos, grp_vals, bottom=bottom,
             color=STACKED_COLORS[grp_name], label=grp_name,
             alpha=0.88, edgecolor="white", lw=0.8, width=0.65)
    bottom += grp_vals

for i, (idx, row) in enumerate(top10_df.iterrows()):
    total = row["Composite"]
    ax7b.text(i, bottom[i] + 0.15, f"{total:.1f}%",
              ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2c3e50")

for i, period in enumerate(top10_df.index):
    period_str = str(period)
    if period_str in EVENT_LABELS:
        event_txt = EVENT_LABELS[period_str]
        bar_top = bottom[i]
        ax7b.annotate(event_txt,
                      xy=(i, bar_top + 0.2),
                      xytext=(i, bar_top + max(bottom) * 0.18 + 0.5),
                      fontsize=7, ha="center", color="#c0392b",
                      fontweight="bold",
                      arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                                      lw=1.2, connectionstyle="arc3,rad=0.1"),
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffeaa7",
                                edgecolor="#c0392b", alpha=0.9))

ax7b.set_xticks(x_pos)
ax7b.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
ax7b.set_title("Top-10 Critical Months — Composite MoM Change (%) by Category",
               fontsize=12, fontweight="bold")
ax7b.set_ylabel("Avg MoM Change (%)", fontsize=10)
ax7b.legend(title="Category", fontsize=9, loc="upper right",
            framealpha=0.9, edgecolor="#bdc3c7")
ax7b.axhline(0, color="gray", lw=0.8)
ax7b.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout(); plt.show()

# ── STL Decomposition ─────────────────────────────────────────────
print("STL Decomposition...")
from statsmodels.tsa.seasonal import STL
for tgt in forecast_targets[:4]:
    col = tgt["col"]; s = tgt["series"].set_index("Date")[col].dropna()
    if len(s) < 24:
        print(f"  Skipping STL for {col} — need 24+ months"); continue
    stl = STL(s, period=12, robust=True).fit()
    fig_stl, axes_stl = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig_stl.suptitle(f"STL Decomposition — {col}", fontsize=12, fontweight="bold")
    for ax, data, label, clr in zip(axes_stl,
        [s, stl.trend, stl.seasonal, stl.resid],
        ["Observed","Trend","Seasonal","Residual"],
        [tgt["color"], "#2c3e50", "#e67e22", "#7f8c8d"]):
        ax.plot(s.index, data, lw=1.5, color=clr)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.show()

# ══════════════════════════════════════════════════════════════════
# PHASE 4 — ALL MODELS × ALL SERIES
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"  PHASE 4 — ALL MODELS × {n} SERIES")
print(f"  Outlier: {OUTLIER_METHOD}  |  Scaler: {SCALER_TYPE}  |  CV folds: {CV_FOLDS}")
print("="*65)

all_accuracy  = []
best_models   = {}
rolling_results = []
cv_results    = []
phase4_preds  = {}

for tgt in forecast_targets:
    m_ = tgt["series"]; col = tgt["col"]; y = m_[col].values; ds = m_["Date"]
    print(f"\n{'─'*60}\n  SERIES: {col}  (n={len(y)})\n{'─'*60}")

    if len(y) < 8:
        print(f"  ⚠  Skipping {col}: only {len(y)} data points"); continue

    train_y  = y[:-FORECAST_MONTHS]
    test_y   = y[-FORECAST_MONTHS:]
    train_ds = ds.iloc[:-FORECAST_MONTHS]
    series_results = []
    phase4_preds[col] = {}

    print(f"  Training size: {len(train_y)}  |  Test size: {len(test_y)}")

    # ── CV ──────────────────────────────────────────────────────
    print(f"\n  [CV] Time-Series Cross Validation ({CV_FOLDS} folds)")
    try:
        cv_mape = cross_validate_sarima(train_y, n_folds=CV_FOLDS)
        if cv_mape is not None:
            cv_results.append({"Series":col, "CV_MAPE":round(cv_mape,2), "Folds":CV_FOLDS})
    except Exception as e: print(f"    CV FAILED: {e}")

    # ── 1 Default SARIMA ────────────────────────────────────────
    print(f"\n  [1] Default SARIMA")
    try:
        fm,_,_,aic = forecast_sarima(train_y, FORECAST_MONTHS)
        m1 = calculate_metrics(test_y, fm, "Default SARIMA")
        if m1["MAPE"] <= 500:
            m1.update({"Model":"SARIMA_Default","Series":col,"AIC":round(aic,1)})
            series_results.append(m1); all_accuracy.append(m1)
            phase4_preds[col]["SARIMA_Default"] = forecast_sarima(y, FORECAST_MONTHS)[0]
    except Exception as e: print(f"    FAILED: {e}")

    # ── 2 GridSearch SARIMA ─────────────────────────────────────
    if USE_GRIDSEARCH:
        print(f"\n  [2] GridSearch SARIMA")
        try:
            gm,_,_,gaic,gord = gridsearch_sarima(train_y, FORECAST_MONTHS)
            m2 = calculate_metrics(test_y, gm, "GridSearch SARIMA")
            if m2["MAPE"] <= 500:
                m2.update({"Model":"SARIMA_GridSearch","Series":col,"AIC":round(gaic,1),"BestOrder":str(gord)})
                series_results.append(m2); all_accuracy.append(m2)
                phase4_preds[col]["SARIMA_GridSearch"] = gridsearch_sarima(y, FORECAST_MONTHS)[0]
        except Exception as e: print(f"    FAILED: {e}")

    # ── 3 RandomSearch SARIMA ───────────────────────────────────
    if USE_RANDOMSEARCH:
        print(f"\n  [3] RandomSearch SARIMA")
        try:
            rm,_,_,raic,rord = randomsearch_sarima(train_y, FORECAST_MONTHS, N_RANDOM)
            m3 = calculate_metrics(test_y, rm, "RandomSearch SARIMA")
            if m3["MAPE"] <= 500:
                m3.update({"Model":"SARIMA_RandomSearch","Series":col,"AIC":round(raic,1),"BestOrder":str(rord)})
                series_results.append(m3); all_accuracy.append(m3)
                phase4_preds[col]["SARIMA_RandomSearch"] = randomsearch_sarima(y, FORECAST_MONTHS, N_RANDOM)[0]
        except Exception as e: print(f"    FAILED: {e}")

    # ── 4 Tuned Prophet ─────────────────────────────────────────
    if USE_PROPHET:
        print(f"\n  [4] Tuned Prophet")
        try:
            fy,_,_ = forecast_prophet(train_ds, train_y, FORECAST_MONTHS, series_name=col)
            m4 = calculate_metrics(test_y, fy, "Prophet")
            if m4["MAPE"] <= 500:
                m4.update({"Model":"Prophet","Series":col,"AIC":"N/A"})
                series_results.append(m4); all_accuracy.append(m4)
                phase4_preds[col]["Prophet"] = forecast_prophet(ds, y, FORECAST_MONTHS, series_name=col)[0]
        except Exception as e: print(f"    FAILED: {e}")

    # ── 5 LSTM ──────────────────────────────────────────────────
    if USE_LSTM:
        print(f"\n  [5] LSTM (scaler={SCALER_TYPE})")
        try:
            lb = max(1, min(12, len(train_y)//4))
            lp,_,_ = forecast_lstm(train_y, FORECAST_MONTHS, look_back=lb)
            if lp is not None:
                m5 = calculate_metrics(test_y, lp, "LSTM")
                if m5["MAPE"] <= 500:
                    m5.update({"Model":"LSTM","Series":col,"AIC":"N/A"})
                    series_results.append(m5); all_accuracy.append(m5)
                    full_lb = max(1, min(12, len(y)//4))
                    full_lp,_,_ = forecast_lstm(y, FORECAST_MONTHS, look_back=full_lb)
                    if full_lp is not None: phase4_preds[col]["LSTM"] = full_lp
        except Exception as e: print(f"    LSTM FAILED: {e}")

    # ── 6 XGBoost ───────────────────────────────────────────────
    if USE_XGBOOST:
        print(f"\n  [6] XGBoost")
        try:
            xp,_,_ = forecast_xgboost(train_y, FORECAST_MONTHS, series_name=col)
            if xp is not None:
                m6 = calculate_metrics(test_y, xp, "XGBoost")
                if m6["MAPE"] <= 500:
                    m6.update({"Model":"XGBoost","Series":col,"AIC":"N/A"})
                    series_results.append(m6); all_accuracy.append(m6)
                    full_xp,_,_ = forecast_xgboost(y, FORECAST_MONTHS, series_name=col)
                    if full_xp is not None: phase4_preds[col]["XGBoost"] = full_xp
        except Exception as e: print(f"    XGBoost FAILED: {e}")

    # ── 7 Random Forest ─────────────────────────────────────────
    if USE_RF:
        print(f"\n  [7] Random Forest")
        try:
            rfp,_,_ = forecast_random_forest(train_y, FORECAST_MONTHS, series_name=col)
            if rfp is not None:
                m7 = calculate_metrics(test_y, rfp, "RandomForest")
                if m7["MAPE"] <= 500:
                    m7.update({"Model":"RandomForest","Series":col,"AIC":"N/A"})
                    series_results.append(m7); all_accuracy.append(m7)
                    full_rfp,_,_ = forecast_random_forest(y, FORECAST_MONTHS, series_name=col)
                    if full_rfp is not None: phase4_preds[col]["RandomForest"] = full_rfp
        except Exception as e: print(f"    RF FAILED: {e}")

    # ── 8 Hybrid SARIMA + LSTM ───────────────────────────────────
    if USE_HYBRID:
        print(f"\n  [8] Hybrid SARIMA+LSTM")
        try:
            lb_hybrid = max(3, min(6, len(train_y)//6))
            hp, hlo, hhi = forecast_hybrid_sarima_lstm(train_y, FORECAST_MONTHS,
                                                        look_back=lb_hybrid)
            if hp is not None:
                m8 = calculate_metrics(test_y, hp, "Hybrid_SARIMA_LSTM")
                if m8["MAPE"] <= 500:
                    m8.update({"Model":"Hybrid_SARIMA_LSTM","Series":col,"AIC":"N/A"})
                    series_results.append(m8); all_accuracy.append(m8)
                    full_hp,_,_ = forecast_hybrid_sarima_lstm(y, FORECAST_MONTHS,
                                                               look_back=lb_hybrid)
                    if full_hp is not None: phase4_preds[col]["Hybrid_SARIMA_LSTM"] = full_hp
        except Exception as e: print(f"    Hybrid FAILED: {e}")

    # ── 9 Rolling Walk-Forward ──────────────────────────────────
    print(f"\n  [9] Rolling Walk-Forward SARIMA")
    try:
        rp, ra, rd = rolling_forecast_sarima(y, ds, FORECAST_MONTHS)
        m9 = calculate_metrics(ra, rp, "Rolling SARIMA")
        if m9["MAPE"] <= 500:
            m9.update({"Model":"Rolling_SARIMA","Series":col,"AIC":"N/A"})
            series_results.append(m9); all_accuracy.append(m9)
            rolling_results.append({"col":col,"clr":tgt["color"],"dates":rd,"actual":ra,"predicted":rp})
    except Exception as e: print(f"    FAILED: {e}")

    # ── Weighted Ensemble ────────────────────────────────────────
    valid_preds = {k: v for k, v in phase4_preds[col].items() if v is not None}
    if len(valid_preds) >= 2:
        mape_lookup = {a["Model"]: a["MAPE"] for a in series_results if "Model" in a}
        weights = {}
        for mn in valid_preds:
            mape_val = mape_lookup.get(mn, 50)
            weights[mn] = 1.0 / max(mape_val, 0.1)
        total_w = sum(weights.values())
        ensemble_pred = np.zeros(FORECAST_MONTHS)
        for mn, pv in valid_preds.items():
            ensemble_pred += (weights[mn] / total_w) * np.array(pv[:FORECAST_MONTHS])
        phase4_preds[col]["Ensemble"] = ensemble_pred

        train_preds_for_ensemble = []
        w_list = []
        for a in series_results:
            mn = a.get("Model",""); mape_v = a.get("MAPE", 50)
            if mape_v > 500: continue
            try:
                if "GridSearch" in mn:
                    tp,_,_,_,_ = gridsearch_sarima(train_y, FORECAST_MONTHS)
                elif "RandomSearch" in mn:
                    tp,_,_,_,_ = randomsearch_sarima(train_y, FORECAST_MONTHS, 5)
                elif "Prophet" in mn:
                    tp,_,_ = forecast_prophet(train_ds, train_y, FORECAST_MONTHS, series_name=col)
                elif "XGBoost" in mn:
                    tp,_,_ = forecast_xgboost(train_y, FORECAST_MONTHS)
                elif "RandomForest" in mn:
                    tp,_,_ = forecast_random_forest(train_y, FORECAST_MONTHS)
                elif "Hybrid" in mn:
                    lb_h = max(3, min(6, len(train_y)//6))
                    tp,_,_ = forecast_hybrid_sarima_lstm(train_y, FORECAST_MONTHS, look_back=lb_h)
                elif "SARIMA_Default" in mn:
                    tp,_,_,_ = forecast_sarima(train_y, FORECAST_MONTHS)
                else:
                    continue
                if tp is not None:
                    train_preds_for_ensemble.append(np.array(tp[:FORECAST_MONTHS]))
                    w_list.append(1.0 / max(mape_v, 0.1))
            except:
                continue

        if len(train_preds_for_ensemble) >= 2:
            tw = sum(w_list)
            ens_train = sum(w * p for w, p in zip(w_list, train_preds_for_ensemble)) / tw
            m_ens = calculate_metrics(test_y, ens_train, "Weighted Ensemble")
            if m_ens["MAPE"] <= 500:
                m_ens.update({"Model":"Ensemble","Series":col,"AIC":"N/A"})
                series_results.append(m_ens); all_accuracy.append(m_ens)
                print(f"    [Weighted Ensemble]  MAPE={m_ens['MAPE']:.2f}%")

    if series_results:
        best = min(series_results, key=lambda x: x["MAPE"]); best_models[col] = best
        print(f"\n  BEST for {col}: {best['Model']}  MAPE={best['MAPE']}%  "
              f"MAE={best['MAE']}  RMSE={best['RMSE']}")

# ══════════════════════════════════════════════════════════════════
# CV SUMMARY CHART
# ══════════════════════════════════════════════════════════════════
if cv_results:
    cv_df = pd.DataFrame(cv_results)
    fig_cv, ax_cv = plt.subplots(figsize=(14,5))
    colors_cv = ["#2ecc71" if v < 10 else "#f39c12" if v < 20 else "#e74c3c"
                 for v in cv_df["CV_MAPE"]]
    bars_cv = ax_cv.bar(range(len(cv_df)), cv_df["CV_MAPE"], color=colors_cv,
                        width=0.6, edgecolor="white")
    ax_cv.set_xticks(range(len(cv_df)))
    ax_cv.set_xticklabels(cv_df["Series"], rotation=30, ha="right", fontsize=9)
    ax_cv.axhline(10, color="green",  lw=1.5, ls="--", alpha=0.7, label="Excellent (10%)")
    ax_cv.axhline(20, color="orange", lw=1.5, ls="--", alpha=0.7, label="Good (20%)")
    ax_cv.set_title(f"Time-Series CV MAPE — SARIMA ({CV_FOLDS} Folds)",
                    fontsize=12, fontweight="bold")
    ax_cv.set_ylabel("CV Mean MAPE (%)")
    for bar, val in zip(bars_cv, cv_df["CV_MAPE"]):
        ax_cv.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax_cv.legend(fontsize=9); ax_cv.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.show()

# ══════════════════════════════════════════════════════════════════
# FORECAST CHARTS
# ══════════════════════════════════════════════════════════════════
print("\n08 Individual forecast charts...")
model_colors = {
    "SARIMA_Default"     : "#e74c3c",
    "SARIMA_GridSearch"  : "#8e44ad",
    "SARIMA_RandomSearch": "#16a085",
    "Prophet"            : "#d35400",
    "LSTM"               : "#c0392b",
    "XGBoost"            : "#1abc9c",
    "RandomForest"       : "#6c5ce7",
    "Rolling_SARIMA"     : "#2980b9",
    "Ensemble"           : "#f39c12",
    "Hybrid_SARIMA_LSTM" : "#00b894",
}

for tgt in forecast_targets:
    m_ = tgt["series"]; col = tgt["col"]; clr = tgt["color"]
    y = m_[col].values; ds = m_["Date"]; last = ds.iloc[-1]
    fd = pd.date_range(last, periods=FORECAST_MONTHS+1, freq="MS")[1:]
    bm_name = best_models.get(col, {}).get("Model", "SARIMA_Default")
    preds_cache = phase4_preds.get(col, {})

    fig_f, ax_top = plt.subplots(figsize=(14, 6))
    fig_f.suptitle(f"{col} — Forecast  |  Best: {bm_name}", fontsize=12, fontweight="bold")

    ax_top.plot(ds, y, color=clr, lw=2.0, label="Historical", zorder=3)

    for mn, pv in preds_cache.items():
        if mn == bm_name: continue
        mc = model_colors.get(mn, "#95a5a6")
        ls = {"Prophet":"-.","XGBoost":":","RandomForest":(0,(3,1,1,1)),
              "LSTM":"--","Ensemble":"-","Hybrid_SARIMA_LSTM":(0,(5,1))}.get(mn, "--")
        ax_top.plot(fd[:len(pv)], pv[:FORECAST_MONTHS], color=mc, lw=1.5, ls=ls,
                    alpha=0.6, label=mn)

    fc_color = model_colors.get(bm_name, "#e74c3c")
    fmean = flo_b = fhi_b = None
    try:
        if bm_name in preds_cache:
            fmean = preds_cache[bm_name]
            flo_b = fmean * 0.92; fhi_b = fmean * 1.08
        if "GridSearch" in bm_name:
            fmean, flo_b, fhi_b, _, _ = gridsearch_sarima(y, FORECAST_MONTHS)
        elif "RandomSearch" in bm_name:
            fmean, flo_b, fhi_b, _, _ = randomsearch_sarima(y, FORECAST_MONTHS, N_RANDOM)
        elif bm_name == "SARIMA_Default":
            fmean, flo_b, fhi_b, _ = forecast_sarima(y, FORECAST_MONTHS)
    except Exception as e:
        print(f"    {col} CI fetch failed: {e}")

    if fmean is not None:
        fmean = np.array(fmean[:FORECAST_MONTHS])
        ax_top.plot(fd, fmean, color=fc_color, lw=3.0, ls="--", marker="o", ms=7,
                    label=f"BEST: {bm_name}", zorder=5)
        if flo_b is not None and fhi_b is not None:
            ax_top.fill_between(fd, np.array(flo_b[:FORECAST_MONTHS]),
                                np.array(fhi_b[:FORECAST_MONTHS]),
                                color=fc_color, alpha=0.18, label="80% CI")
        for date_p, val_p in zip(fd, fmean):
            ax_top.annotate(f"₹{val_p:.1f}", (date_p, val_p),
                            textcoords="offset points", xytext=(0, 10),
                            fontsize=7, color=fc_color, ha="center", fontweight="bold")

    enr = tgt["enriched"]
    for _, fr in enr[(enr["Date"] >= ds.iloc[-min(12,len(ds))]) & (enr["Festival"]==1)].iterrows():
        ax_top.axvline(fr["Date"], color="#d63031", lw=1.2, ls=":", alpha=0.7)
        ax_top.text(fr["Date"], ax_top.get_ylim()[1]*0.95, fr["Festival_Label"],
                    rotation=90, fontsize=7, color="#d63031", va="top")

    ax_top.axvline(last, color="gray", lw=1.5, ls=":", alpha=0.8, label="Forecast Start")
    ax_top.set_ylabel("Price (₹)", fontsize=9)
    ax_top.legend(fontsize=8, loc="upper left", ncol=2)
    fmt_ax(ax_top)
    plt.tight_layout(); plt.show()

# ══════════════════════════════════════════════════════════════════
# ROLLING FORECAST CHART
# ══════════════════════════════════════════════════════════════════
if rolling_results:
    print("09 Rolling forecast chart...")
    fig_r, axes_r = plt.subplots(len(rolling_results), 1,
                                  figsize=(13, 5*len(rolling_results)))
    if len(rolling_results) == 1: axes_r = [axes_r]
    fig_r.suptitle("Rolling Walk-Forward Forecast vs Actual",
                   fontsize=13, fontweight="bold")
    for ax, res in zip(axes_r, rolling_results):
        ax.plot(res["dates"], res["actual"],    color=res["clr"], lw=2.0,
                marker="o", ms=6, label="Actual")
        ax.plot(res["dates"], res["predicted"], color="#e74c3c", lw=2.0,
                marker="s", ms=6, ls="--", label="Predicted")
        ax.fill_between(res["dates"], res["actual"], res["predicted"],
                        alpha=0.15, color="gray", label="Error Area")
        mr = calculate_metrics(res["actual"], res["predicted"])
        ax.set_title(f"{res['col']} — Rolling  MAE={mr['MAE']:.2f}  MAPE={mr['MAPE']:.2f}%",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel(res["col"], fontsize=9); ax.legend(fontsize=9); fmt_ax(ax, interval=1)
    plt.tight_layout(); plt.show()

# ══════════════════════════════════════════════════════════════════
# MASTER ACCURACY TABLE
# ══════════════════════════════════════════════════════════════════
# MAIN SLIDE — Summary: Best MAPE + Best Model per series (bar chart)
# APPENDIX  — Full MAPE table + heatmap
# ══════════════════════════════════════════════════════════════════
if all_accuracy:
    print("\nAccuracy tables...")
    acc_df = pd.DataFrame(all_accuracy)
    cols_o = [c for c in ["Series","Model","MAE","RMSE","MAPE","SMAPE","AIC"]
              if c in acc_df.columns]
    acc_df = acc_df[cols_o].sort_values(["Series","MAPE"]).reset_index(drop=True)
    acc_df.to_csv(os.path.join(WORK_DIR, "model_accuracy_all.csv"), index=False)

    # Build pivot used throughout
    pivot = acc_df.pivot_table(index="Series", columns="Model", values="MAPE", aggfunc="min")
    model_order = sorted(pivot.columns.tolist())
    pivot = pivot[model_order]
    pivot["Best_MAPE"]  = pivot[model_order].min(axis=1).round(2)
    pivot["Best_Model"] = pivot[model_order].idxmin(axis=1)

    # ─────────────────────────────────────────────────────────────
    # MAIN SLIDE: Summary bar chart — Best MAPE per series
    # ─────────────────────────────────────────────────────────────
    print("  [MAIN SLIDE] Summary — Best MAPE + Best Model per series")
    summary_df = pivot[["Best_MAPE","Best_Model"]].sort_values("Best_MAPE").reset_index()

    bar_clrs = ["#2ecc71" if v < 10 else "#f39c12" if v < 20 else "#e74c3c"
                for v in summary_df["Best_MAPE"]]

    fig_sum, ax_sum = plt.subplots(figsize=(14, max(5, len(summary_df)*0.55 + 2)),
                                    facecolor="#f8f9fa")
    ax_sum.set_facecolor("#fdfdfd")

    bars = ax_sum.barh(summary_df["Series"], summary_df["Best_MAPE"],
                       color=bar_clrs, edgecolor="white", lw=0.8, height=0.65)

    # Annotate: value + best model name on each bar
    for bar, (_, row) in zip(bars, summary_df.iterrows()):
        w = bar.get_width()
        ax_sum.text(w + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{w:.1f}%  [{row['Best_Model']}]",
                    va="center", fontsize=9, fontweight="bold", color="#2c3e50")

    ax_sum.axvline(10, color="#2ecc71", lw=1.5, ls="--", alpha=0.7, label="Excellent < 10%")
    ax_sum.axvline(20, color="#f39c12", lw=1.5, ls="--", alpha=0.7, label="Good < 20%")

    # Legend patches for colour coding
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Excellent  < 10%"),
        mpatches.Patch(color="#f39c12", label="Good       10–20%"),
        mpatches.Patch(color="#e74c3c", label="Needs work > 20%"),
    ]
    ax_sum.legend(handles=legend_patches, fontsize=9, loc="lower right",
                  framealpha=0.9, edgecolor="#bdc3c7")

    ax_sum.set_xlabel("Best MAPE (%)", fontsize=10)
    ax_sum.set_title("Model Accuracy Summary — Best MAPE & Winning Model per Series\n"
                     "(lower MAPE = better  |  bar label shows winning model)",
                     fontsize=12, fontweight="bold")
    ax_sum.set_xlim(0, summary_df["Best_MAPE"].max() * 1.45)
    ax_sum.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────────────────────────────
    # APPENDIX A: Full MAPE table (all models × all series)
    # ─────────────────────────────────────────────────────────────
    print("  [APPENDIX A] Full MAPE table")
    n_rows = len(pivot); display_cols = model_order + ["Best_MAPE","Best_Model"]
    n_cols = len(display_cols)

    fig_acc, ax_acc = plt.subplots(
        figsize=(max(16, n_cols*2.0), max(6, n_rows*0.9 + 3)))
    ax_acc.axis("off")
    fig_acc.suptitle(
        "APPENDIX A — Full Model Accuracy Table  |  MAPE (%)\n"
        "★ = Best per series  •  Green < 10%  •  Orange 10–20%  •  Red > 20%  |  "
        "Blue cols = Summary",
        fontsize=10, fontweight="bold", y=0.98)

    cell_text   = []
    cell_colors = []

    for series_name, row in pivot.iterrows():
        best_val   = row["Best_MAPE"]
        best_model = row["Best_Model"]
        row_text   = []
        row_colors = []
        for mn in display_cols:
            if mn == "Best_Model":
                row_text.append(str(best_model))
                row_colors.append("#d6eaf8")
            elif mn == "Best_MAPE":
                row_text.append(f"{best_val:.1f}%")
                row_colors.append("#d6eaf8")
            else:
                val = row.get(mn, np.nan)
                if pd.isna(val):
                    row_text.append("—"); row_colors.append("#f2f3f4")
                else:
                    star = " ★" if abs(val - best_val) < 0.01 else ""
                    row_text.append(f"{val:.1f}%{star}")
                    if   val < 10:  row_colors.append("#d5f5e3")
                    elif val < 20:  row_colors.append("#fdebd0")
                    elif val < 50:  row_colors.append("#fadbd8")
                    else:           row_colors.append("#c0392b")
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    tbl = ax_acc.table(
        cellText    = cell_text,
        rowLabels   = list(pivot.index),
        colLabels   = display_cols,
        cellColours = cell_colors,
        cellLoc     = "center",
        loc         = "center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.2, 2.2)

    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1a5276" if display_cols[j] in ["Best_MAPE","Best_Model"] else "#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7)
    for i in range(1, n_rows+1):
        cell = tbl[i, -1]
        cell.set_facecolor("#eaf2ff")
        cell.set_text_props(fontweight="bold", fontsize=8)

    plt.tight_layout(); plt.show()

    # ─────────────────────────────────────────────────────────────
    # APPENDIX B: MAPE Heatmap
    # ─────────────────────────────────────────────────────────────
    print("  [APPENDIX B] MAPE Heatmap")
    pivot_heat = acc_df.pivot_table(index="Series", columns="Model",
                                     values="MAPE", aggfunc="min")[model_order].copy()
    pivot_heat["_best"] = pivot_heat.min(axis=1)
    pivot_heat = pivot_heat.sort_values("_best").drop(columns="_best")
    n_rows_h = len(pivot_heat); n_cols_h = len(model_order)

    display_vals = pivot_heat.values.astype(float)
    capped_vals  = np.clip(display_vals, 0, 30)
    best_mapes_h       = pivot_heat.min(axis=1)
    best_models_heat_h = pivot_heat.idxmin(axis=1)

    fig_heat, ax_heat = plt.subplots(
        figsize=(max(12, n_cols_h*1.8), max(5, n_rows_h*0.8 + 2)))
    fig_heat.suptitle(
        "APPENDIX B — MAPE Heatmap  (lower = better)\n"
        "Color capped at 30%  |  ★ = Best model per series  |  Values > 50% in dark red",
        fontsize=11, fontweight="bold")

    im = ax_heat.imshow(capped_vals, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=30)
    ax_heat.set_xticks(range(n_cols_h))
    ax_heat.set_xticklabels(model_order, rotation=40, ha="right", fontsize=9)
    ax_heat.set_yticks(range(n_rows_h))
    ax_heat.set_yticklabels(list(pivot_heat.index), fontsize=9)
    fig_heat.colorbar(im, ax=ax_heat, shrink=0.8, label="MAPE (%) — capped at 30")

    for i, series_name in enumerate(pivot_heat.index):
        b_mod = best_models_heat_h[series_name]
        for j, mn in enumerate(model_order):
            val = pivot_heat.loc[series_name, mn]
            if not pd.isna(val):
                star = "★" if mn == b_mod else ""
                cell_txt = f"{star}{val:.1f}%"
                if val > 50:
                    rect = plt.Rectangle([j-0.5, i-0.5], 1, 1, color="#6e2c00", zorder=2)
                    ax_heat.add_patch(rect)
                    txt_color = "white"
                elif val > 30: txt_color = "white"
                else:          txt_color = "white" if val > 20 else "black"
                ax_heat.text(j, i, cell_txt, ha="center", va="center",
                             fontsize=8, fontweight="bold", color=txt_color, zorder=3)

    for i, series_name in enumerate(pivot_heat.index):
        ax_heat.text(n_cols_h + 0.1, i,
                     f"Best: {best_models_heat_h[series_name]} ({best_mapes_h[series_name]:.1f}%)",
                     va="center", fontsize=8, color="#1a5276", fontweight="bold")

    ax_heat.set_xlim(-0.5, n_cols_h + 3.5)
    plt.tight_layout(); plt.show()

print("\n" + "="*65)
print("  COMPLETE")
print("="*65)