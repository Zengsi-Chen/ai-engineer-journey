import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def validate_price_data(df: pd.DataFrame, ticker: str) -> dict:
    errors = []
    warnings = []

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    if errors:
        return {
            "ticker": ticker,
            "status": "FAILED",
            "errors": errors,
            "warnings": warnings,
        }

    if df.empty:
        errors.append("DataFrame is empty")

        return {
            "ticker": ticker,
            "status": "FAILED",
            "errors": errors,
            "warnings": warnings,
        }

    # ---------------------------------------------------------
    # Basic structure checks
    # ---------------------------------------------------------

    duplicate_dates = int(
        df["Date"].duplicated().sum()
    )

    if duplicate_dates > 0:
        errors.append(
            f"Duplicate dates: {duplicate_dates}"
        )

    missing_values = (
        df[REQUIRED_COLUMNS]
        .isna()
        .sum()
        .to_dict()
    )

    missing_total = sum(missing_values.values())

    if missing_total > 0:
        errors.append(
            f"Missing values: {missing_values}"
        )

    # ---------------------------------------------------------
    # Price validity
    # ---------------------------------------------------------

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    non_positive = {}

    for column in price_columns:
        count = int(
            (df[column] <= 0).sum()
        )

        if count > 0:
            non_positive[column] = count

    if non_positive:
        errors.append(
            f"Non-positive prices: {non_positive}"
        )

    # ---------------------------------------------------------
    # OHLC consistency
    # ---------------------------------------------------------

    invalid_high_low = (
        df["High"] < df["Low"]
    )

    invalid_open_range = (
        (df["Open"] > df["High"])
        | (df["Open"] < df["Low"])
    )

    invalid_close_range = (
        (df["Close"] > df["High"])
        | (df["Close"] < df["Low"])
    )

    # ---------------------------------------------------------
    # Historical vs latest-session anomalies
    # ---------------------------------------------------------

    latest_date = df["Date"].max()

    historical_open_errors = (
        invalid_open_range
        & (df["Date"] < latest_date)
    )

    latest_open_errors = (
        invalid_open_range
        & (df["Date"] == latest_date)
    )

    historical_high_low_errors = (
        invalid_high_low
        & (df["Date"] < latest_date)
    )

    latest_high_low_errors = (
        invalid_high_low
        & (df["Date"] == latest_date)
    )

    historical_close_errors = (
        invalid_close_range
        & (df["Date"] < latest_date)
    )

    latest_close_errors = (
        invalid_close_range
        & (df["Date"] == latest_date)
    )

    # Historical anomalies are hard failures.

    if historical_high_low_errors.any():
        errors.append(
            "Historical High < Low: "
            f"{int(historical_high_low_errors.sum())}"
        )

    if historical_open_errors.any():
        errors.append(
            "Historical Open outside High/Low range: "
            f"{int(historical_open_errors.sum())}"
        )

    if historical_close_errors.any():
        errors.append(
            "Historical Close outside High/Low range: "
            f"{int(historical_close_errors.sum())}"
        )

    # Latest-session anomalies are warnings.

    if latest_high_low_errors.any():
        warnings.append(
            "Latest session High < Low: "
            f"{int(latest_high_low_errors.sum())}"
        )

    if latest_open_errors.any():
        warnings.append(
            "Latest session Open outside High/Low range: "
            f"{int(latest_open_errors.sum())}"
        )

    if latest_close_errors.any():
        warnings.append(
            "Latest session Close outside High/Low range: "
            f"{int(latest_close_errors.sum())}"
        )

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    negative_volume = int(
        (df["Volume"] < 0).sum()
    )

    if negative_volume > 0:
        errors.append(
            f"Negative volume: {negative_volume}"
        )

    zero_volume = int(
        (df["Volume"] == 0).sum()
    )

    if zero_volume > 0:
        warnings.append(
            f"Zero-volume sessions: {zero_volume}"
        )

    # ---------------------------------------------------------
    # Date ordering
    # ---------------------------------------------------------

    if not df["Date"].is_monotonic_increasing:
        errors.append(
            "Dates are not sorted ascending"
        )

    # ---------------------------------------------------------
    # Observation count
    # ---------------------------------------------------------

    observations = len(df)

    if observations < 400:
        warnings.append(
            f"Only {observations} observations"
        )

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    if errors:
        status = "FAILED"
    elif warnings:
        status = "PASSED_WITH_WARNING"
    else:
        status = "PASSED"

    return {
        "ticker": ticker,
        "status": status,
        "observations": observations,
        "start": df["Date"].min(),
        "end": df["Date"].max(),
        "errors": errors,
        "warnings": warnings,
    }