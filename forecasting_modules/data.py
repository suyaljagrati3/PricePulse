"""Dataset loading and common forecast-target construction."""

from pathlib import Path
import pandas as pd

from .features import add_features, remove_outliers

GROCERY_COLUMNS = ["Tea", "Salt", "Mustard", "Palm", "Wheat", "Milk", "Sugar", "Gur", "Gram", "Moong", "Soya", "Tur"]
VEGETABLE_COLUMNS = ["Onion_Price", "Potato_Price", "Tomato_Price"]
COLORS = ["#2ecc71", "#e67e22", "#3498db", "#9b59b6", "#f39c12", "#e74c3c", "#1abc9c", "#e91e63", "#00bcd4", "#8bc34a"]


def to_monthly(df, column, aggregation="mean"):
    data = df.copy()
    data["YM"] = data["Date"].dt.to_period("M")
    monthly = data.groupby("YM")[column].agg(aggregation).reset_index()
    monthly["Date"] = monthly["YM"].dt.to_timestamp()
    return monthly.sort_values("Date").reset_index(drop=True)


def _read_csv(path, date_format=None, column_renames=None):
    data = pd.read_csv(path)
    data.columns = [column.strip() for column in data.columns]
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    if column_renames:
        data = data.rename(columns={old: new for old, new in column_renames.items() if old in data.columns and new not in data.columns})
    data["Date"] = pd.to_datetime(data["Date"], format=date_format, dayfirst=date_format is None, errors="coerce")
    return data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def load_source_data(data_dir="datasets"):
    """Load and normalize source files. Returned keys match the legacy pipeline."""
    root = Path(data_dir)
    fuel = _read_csv(
        root / "fuel_by_date.csv",
        column_renames={"date": "Date", "rate": "Fuel_Price", "Rate": "Fuel_Price", "fuel_price": "Fuel_Price", "price": "Fuel_Price"},
    )
    vegetables = _read_csv(root / "vegetable_inflation_dataset.csv", "%Y-%m-%d")
    price_column = "Average_Price" if "Average_Price" in vegetables else "Average"
    if price_column in vegetables:
        vegetables = vegetables.rename(columns={price_column: "Veg_Price"})
    elif "Veg_Price" not in vegetables:
        numeric = vegetables.select_dtypes(include="number").columns
        if not len(numeric):
            raise ValueError("Vegetable dataset contains no numeric price column.")
        vegetables = vegetables.rename(columns={numeric[0]: "Veg_Price"})
    grocery = _read_csv(root / "merged_grocery_dataset.csv")
    commodity_frames = {}
    for column in GROCERY_COLUMNS + VEGETABLE_COLUMNS:
        source = grocery if column in grocery.columns else vegetables
        if column in source.columns:
            commodity_frames[column] = to_monthly(source[["Date", column]].dropna(), column)
    return {"fuel_m": to_monthly(fuel, "Fuel_Price"), "veg_m": to_monthly(vegetables[["Date", "Veg_Price"]], "Veg_Price"), "commodity_dfs": commodity_frames}


def build_forecast_targets(data_dir="datasets", outlier_method="iqr"):
    sources = load_source_data(data_dir)
    targets = [("Veg_Price", sources["veg_m"]), ("Fuel_Price", sources["fuel_m"])] + list(sources["commodity_dfs"].items())
    result = []
    for index, (column, frame) in enumerate(targets):
        clean = frame.copy()
        clean[column], _ = remove_outliers(clean[column], outlier_method)
        result.append({"label": column, "series": clean, "col": column, "color": COLORS[index % len(COLORS)], "enriched": add_features(clean, column)})
    return result
