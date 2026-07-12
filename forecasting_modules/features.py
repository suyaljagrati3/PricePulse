"""Feature engineering and data-cleaning helpers."""

import numpy as np

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FESTIVAL_MONTHS = {
    (2019, 10): "Diwali", (2020, 10): "Diwali", (2021, 11): "Diwali",
    (2022, 10): "Diwali", (2023, 11): "Diwali", (2020, 3): "Lockdown1",
    (2020, 4): "Lockdown1", (2020, 5): "Eid", (2021, 4): "Lockdown2",
    (2021, 5): "Eid", (2020, 1): "COVID_Start", (2019, 8): "Eid",
    (2022, 5): "Eid", (2023, 4): "Eid",
}


def get_season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "PostMonsoon"


def add_features(df, price_col, lags=(1, 2, 3, 6, 12)):
    """Return a date-sorted frame enriched with calendar and lag features."""
    data = df.copy().sort_values("Date").reset_index(drop=True)
    data["Month"] = data["Date"].dt.month
    data["Month_Name"] = data["Month"].map(lambda value: MONTH_LABELS[value - 1])
    data["Year"] = data["Date"].dt.year
    data["Season"] = data["Month"].map(get_season)
    data["Season_Code"] = data["Season"].map({"Winter": 0, "Summer": 1, "Monsoon": 2, "PostMonsoon": 3})
    data["Festival"] = data.apply(lambda row: int((row["Year"], row["Month"]) in FESTIVAL_MONTHS), axis=1)
    data["Festival_Label"] = data.apply(lambda row: FESTIVAL_MONTHS.get((row["Year"], row["Month"]), ""), axis=1)
    for lag in lags:
        data[f"Lag_{lag}"] = data[price_col].shift(lag)
    for window in (3, 6):
        data[f"Roll_Mean_{window}"] = data[price_col].rolling(window, min_periods=1).mean()
        data[f"Roll_Std_{window}"] = data[price_col].rolling(window, min_periods=1).std().fillna(0)
    data["MoM_Pct"] = data[price_col].pct_change() * 100
    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12)
    return data


def remove_outliers(series, method="iqr"):
    """Interpolate IQR- or z-score-based outliers and return values plus mask."""
    cleaned = series.copy().astype(float)
    if method == "iqr":
        first, third = cleaned.quantile(0.25), cleaned.quantile(0.75)
        spread = third - first
        mask = (cleaned < first - 1.5 * spread) | (cleaned > third + 1.5 * spread)
    else:
        mask = ((cleaned - cleaned.mean()) / cleaned.std()).abs() > 3
    cleaned[mask] = np.nan
    return cleaned.interpolate(method="linear").ffill().bfill(), mask
