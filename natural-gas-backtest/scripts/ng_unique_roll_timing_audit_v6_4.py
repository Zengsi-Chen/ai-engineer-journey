from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================================
# NG UNIQUE MARKET ROLL EVENT AUDIT V6.4
#
# Purpose:
#   Deduplicate V6.3 strategy-weighted roll records into UNIQUE market
#   roll events.
#
# Key principle:
#   One real market roll event = one observation.
#
#   Multiple strategy parameter combinations may reference the same market
#   roll event. Those duplicate strategy records are retained as metadata,
#   but are NOT counted multiple times in the unique-event statistics.
# ============================================================================


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

INPUT_FILE = DATA_DIR / "ng_economic_roll_attribution_v6_2_rolls.csv"

EVENT_FILE = DATA_DIR / "ng_unique_roll_timing_audit_v6_4_events.csv"
SUMMARY_FILE = DATA_DIR / "ng_unique_roll_timing_audit_v6_4_summary.csv"
YEARLY_FILE = DATA_DIR / "ng_unique_roll_timing_audit_v6_4_yearly.csv"
DELAY_FILE = DATA_DIR / "ng_unique_roll_timing_audit_v6_4_delay.csv"
CONCENTRATION_FILE = DATA_DIR / "ng_unique_roll_timing_audit_v6_4_concentration.csv"
REMOVE_ONE_FILE = DATA_DIR / "ng_unique_roll_timing_audit_v6_4_remove_one_event.csv"
INTEGRITY_FILE = DATA_DIR / "ng_unique_roll_timing_audit_v6_4_integrity.csv"


# Numerical tolerance for duplicate-event consistency checks.
TOL = 1e-9


REQUIRED_COLUMNS = [
    "trade_id",
    "entry_threshold",
    "exit_threshold",
    "roll_number",
    "old_contract",
    "new_contract",
    "actual_roll_date",
    "actual_old_exit_price",
    "actual_new_entry_price",
    "actual_spread_points",
    "actual_transition_pnl",
    "deferred_old_last_date",
    "deferred_new_entry_date",
    "delay_days",
    "deferred_old_last_close",
    "deferred_new_entry_price",
    "deferred_spread_points",
    "deferred_transition_pnl",
    "transition_effect_deferred_minus_actual",
    "deferred_status",
]


# ============================================================================
# HELPERS
# ============================================================================

def fmt_money(x):
    return f"${x:,.0f}"


def fmt_points(x):
    return f"{x:.6f}"


def normalize_date(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def safe_join(values):
    vals = sorted({str(v) for v in values if pd.notna(v)})
    return ";".join(vals)


def safe_join_float(values):
    vals = sorted({float(v) for v in values if pd.notna(v)})
    return ";".join(f"{v:g}" for v in vals)


def max_range(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.max() - s.min())


# ============================================================================
# LOAD
# ============================================================================

print("=" * 78)
print("NG UNIQUE MARKET ROLL EVENT AUDIT V6.4")
print("=" * 78)

print("\nINPUT")
print("-" * 78)
print(f"Roll attribution : {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded      : {len(df)}")

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(f"  - {c}" for c in missing)
    )

# Normalize dates.
for col in [
    "actual_roll_date",
    "deferred_old_last_date",
    "deferred_new_entry_date",
]:
    df[col] = normalize_date(df[col])

# Numeric columns.
numeric_cols = [
    "entry_threshold",
    "exit_threshold",
    "roll_number",
    "actual_old_exit_price",
    "actual_new_entry_price",
    "actual_spread_points",
    "actual_transition_pnl",
    "delay_days",
    "deferred_old_last_close",
    "deferred_new_entry_price",
    "deferred_spread_points",
    "deferred_transition_pnl",
    "transition_effect_deferred_minus_actual",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================================
# FILTER VALID DEFERRED RECORDS
# ============================================================================

print("\nSTATUS FILTER")
print("-" * 78)

status_counts = df["deferred_status"].value_counts(dropna=False)

for status, count in status_counts.items():
    print(f"{str(status):35s} {count:5d}")

valid = df[
    df["deferred_status"].eq("FULLY_RECONCILED")
].copy()

invalid = df[
    ~df["deferred_status"].eq("FULLY_RECONCILED")
].copy()

print(f"\nTotal records                  : {len(df)}")
print(f"Fully reconciled records      : {len(valid)}")
print(f"Excluded infeasible/other     : {len(invalid)}")


if len(valid) == 0:
    raise ValueError("No FULLY_RECONCILED records available.")


# ============================================================================
# UNIQUE MARKET EVENT KEY
#
# We deliberately include:
#
#   old_contract
#   new_contract
#   actual_roll_date
#   deferred_new_entry_date
#
# This identifies the actual market roll and its corresponding deferred
# entry date.
#
# deferred_old_last_date is also retained and checked below.
# ============================================================================

valid["unique_event_key"] = (
    valid["old_contract"].astype(str)
    + "|"
    + valid["new_contract"].astype(str)
    + "|"
    + valid["actual_roll_date"].astype(str)
    + "|"
    + valid["deferred_new_entry_date"].astype(str)
)


# ============================================================================
# INTEGRITY CHECKS
# ============================================================================

print("\nDUPLICATE EVENT INTEGRITY")
print("-" * 78)

grouped = valid.groupby("unique_event_key", sort=False)

integrity_rows = []

for key, g in grouped:

    def rng(col):
        return max_range(g[col])

    row = {
        "unique_event_key": key,
        "strategy_trade_count": len(g),
        "trade_id_count": g["trade_id"].nunique(),
        "old_contract_count": g["old_contract"].nunique(),
        "new_contract_count": g["new_contract"].nunique(),
        "actual_roll_date_count": g["actual_roll_date"].nunique(),
        "deferred_old_last_date_count": g["deferred_old_last_date"].nunique(),
        "deferred_new_entry_date_count": g["deferred_new_entry_date"].nunique(),
        "delay_range": rng("delay_days"),
        "actual_spread_range": rng("actual_spread_points"),
        "deferred_spread_range": rng("deferred_spread_points"),
        "actual_transition_pnl_range": rng("actual_transition_pnl"),
        "deferred_transition_pnl_range": rng("deferred_transition_pnl"),
        "timing_effect_range": rng(
            "transition_effect_deferred_minus_actual"
        ),
    }

    row["integrity_ok"] = (
        row["old_contract_count"] == 1
        and row["new_contract_count"] == 1
        and row["actual_roll_date_count"] == 1
        and row["deferred_old_last_date_count"] == 1
        and row["deferred_new_entry_date_count"] == 1
        and row["delay_range"] <= TOL
        and row["actual_spread_range"] <= TOL
        and row["deferred_spread_range"] <= TOL
        and row["actual_transition_pnl_range"] <= TOL
        and row["deferred_transition_pnl_range"] <= TOL
        and row["timing_effect_range"] <= TOL
    )

    integrity_rows.append(row)

integrity = pd.DataFrame(integrity_rows)

bad_integrity = integrity[
    ~integrity["integrity_ok"]
].copy()

print(f"Strategy roll records : {len(valid)}")
print(f"Unique market events  : {len(integrity)}")
print(
    f"Integrity failures    : {len(bad_integrity)}"
)

if len(bad_integrity) > 0:
    print("\nWARNING: duplicate market events are not internally identical.")

    print(
        bad_integrity[
            [
                "unique_event_key",
                "strategy_trade_count",
                "delay_range",
                "actual_spread_range",
                "deferred_spread_range",
                "actual_transition_pnl_range",
                "deferred_transition_pnl_range",
                "timing_effect_range",
            ]
        ].to_string(index=False)
    )


# ============================================================================
# BUILD UNIQUE EVENT TABLE
# ============================================================================

unique_rows = []

for key, g in grouped:

    first = g.iloc[0]

    timing_effect = float(
        first["transition_effect_deferred_minus_actual"]
    )

    actual_transition = float(
        first["actual_transition_pnl"]
    )

    deferred_transition = float(
        first["deferred_transition_pnl"]
    )

    row = {
        "unique_event_key": key,

        "old_contract": first["old_contract"],
        "new_contract": first["new_contract"],

        "actual_roll_date": first["actual_roll_date"],
        "deferred_old_last_date": first["deferred_old_last_date"],
        "deferred_new_entry_date": first["deferred_new_entry_date"],

        "delay_days": float(first["delay_days"]),

        "actual_old_exit_price": float(
            first["actual_old_exit_price"]
        ),
        "actual_new_entry_price": float(
            first["actual_new_entry_price"]
        ),
        "actual_spread_points": float(
            first["actual_spread_points"]
        ),
        "actual_transition_pnl": actual_transition,

        "deferred_old_last_close": float(
            first["deferred_old_last_close"]
        ),
        "deferred_new_entry_price": float(
            first["deferred_new_entry_price"]
        ),
        "deferred_spread_points": float(
            first["deferred_spread_points"]
        ),
        "deferred_transition_pnl": deferred_transition,

        "timing_effect_points": (
            float(first["deferred_spread_points"])
            - float(first["actual_spread_points"])
        ),

        "timing_effect_pnl": timing_effect,

        "strategy_trade_count": len(g),

        "trade_ids": safe_join(g["trade_id"]),

        "entry_thresholds": safe_join_float(
            g["entry_threshold"]
        ),

        "exit_thresholds": safe_join_float(
            g["exit_threshold"]
        ),

        "roll_numbers": safe_join_float(
            g["roll_number"]
        ),

        "integrity_ok": bool(
            integrity.loc[
                integrity["unique_event_key"] == key,
                "integrity_ok"
            ].iloc[0]
        ),
    }

    unique_rows.append(row)


events = pd.DataFrame(unique_rows)

events["actual_roll_date"] = pd.to_datetime(
    events["actual_roll_date"]
)

events["deferred_old_last_date"] = pd.to_datetime(
    events["deferred_old_last_date"]
)

events["deferred_new_entry_date"] = pd.to_datetime(
    events["deferred_new_entry_date"]
)

events["year"] = events["actual_roll_date"].dt.year


# ============================================================================
# BASIC STATISTICS
# ============================================================================

n_strategy_records = len(valid)
n_unique_events = len(events)

total_timing = events["timing_effect_pnl"].sum()

positive = events[
    events["timing_effect_pnl"] > TOL
]

negative = events[
    events["timing_effect_pnl"] < -TOL
]

neutral = events[
    events["timing_effect_pnl"].abs() <= TOL
]

positive_count = len(positive)
negative_count = len(negative)
neutral_count = len(neutral)

mean_timing = events["timing_effect_pnl"].mean()
median_timing = events["timing_effect_pnl"].median()
std_timing = events["timing_effect_pnl"].std(ddof=1)

positive_share = (
    positive_count / n_unique_events
    if n_unique_events
    else np.nan
)

duplication_factor = (
    n_strategy_records / n_unique_events
    if n_unique_events
    else np.nan
)


# ============================================================================
# SUMMARY
# ============================================================================

summary_rows = [
    ("input_strategy_roll_records", len(df)),
    ("excluded_non_reconciled_records", len(invalid)),
    ("valid_strategy_roll_records", n_strategy_records),
    ("unique_market_roll_events", n_unique_events),
    ("strategy_records_per_unique_event", duplication_factor),

    ("unique_total_timing_effect_pnl", total_timing),
    ("unique_mean_timing_effect_pnl", mean_timing),
    ("unique_median_timing_effect_pnl", median_timing),
    ("unique_std_timing_effect_pnl", std_timing),

    ("unique_positive_events", positive_count),
    ("unique_negative_events", negative_count),
    ("unique_neutral_events", neutral_count),

    ("unique_positive_share", positive_share),

    ("unique_mean_delay_days", events["delay_days"].mean()),
    ("unique_median_delay_days", events["delay_days"].median()),
    ("unique_max_delay_days", events["delay_days"].max()),

    ("integrity_failure_events", len(bad_integrity)),
]

summary = pd.DataFrame(
    summary_rows,
    columns=["metric", "value"]
)


# ============================================================================
# YEARLY UNIQUE EVENTS
# ============================================================================

yearly_rows = []

for year, g in events.groupby("year"):

    p = g[g["timing_effect_pnl"] > TOL]
    n = g[g["timing_effect_pnl"] < -TOL]

    yearly_rows.append({
        "year": int(year),
        "unique_events": len(g),
        "timing_effect_pnl": g["timing_effect_pnl"].sum(),
        "mean_timing_effect_pnl": g["timing_effect_pnl"].mean(),
        "median_timing_effect_pnl": g["timing_effect_pnl"].median(),
        "positive_events": len(p),
        "negative_events": len(n),
        "neutral_events": len(g) - len(p) - len(n),
        "positive_share": (
            len(p) / len(g)
            if len(g)
            else np.nan
        ),
        "mean_delay_days": g["delay_days"].mean(),
        "median_delay_days": g["delay_days"].median(),
        "max_delay_days": g["delay_days"].max(),
    })

yearly = pd.DataFrame(yearly_rows).sort_values("year")


# ============================================================================
# DELAY BUCKETS
# ============================================================================

def delay_bucket(x):
    if x <= 2:
        return "0-2 days"
    elif x <= 4:
        return "3-4 days"
    elif x <= 6:
        return "5-6 days"
    else:
        return "7+ days"


events["delay_bucket"] = events["delay_days"].apply(delay_bucket)

delay_rows = []

for bucket, g in events.groupby(
    "delay_bucket",
    sort=False
):

    p = g[g["timing_effect_pnl"] > TOL]
    n = g[g["timing_effect_pnl"] < -TOL]

    delay_rows.append({
        "delay_bucket": bucket,
        "unique_events": len(g),
        "timing_effect_pnl": g["timing_effect_pnl"].sum(),
        "mean_timing_effect_pnl": g["timing_effect_pnl"].mean(),
        "median_timing_effect_pnl": g["timing_effect_pnl"].median(),
        "positive_events": len(p),
        "negative_events": len(n),
        "positive_share": (
            len(p) / len(g)
            if len(g)
            else np.nan
        ),
        "mean_delay_days": g["delay_days"].mean(),
    })

delay = pd.DataFrame(delay_rows)


# ============================================================================
# CONCENTRATION
# ============================================================================

sorted_events = events.sort_values(
    "timing_effect_pnl",
    ascending=False
).reset_index(drop=True)

concentration_rows = []

for n in [1, 3, 5, 10]:

    k = min(n, len(sorted_events))

    contribution = (
        sorted_events
        .head(k)["timing_effect_pnl"]
        .sum()
    )

    share = (
        contribution / total_timing
        if total_timing != 0
        else np.nan
    )

    concentration_rows.append({
        "top_n": n,
        "events_included": k,
        "timing_effect_pnl": contribution,
        "share_of_total_timing_effect": share,
    })


concentration = pd.DataFrame(concentration_rows)


# ============================================================================
# REMOVE-ONE-EVENT ANALYSIS
# ============================================================================

remove_rows = []

for _, row in events.iterrows():

    effect = float(row["timing_effect_pnl"])

    remaining = total_timing - effect

    share = (
        effect / total_timing
        if total_timing != 0
        else np.nan
    )

    remove_rows.append({
        "unique_event_key": row["unique_event_key"],
        "old_contract": row["old_contract"],
        "new_contract": row["new_contract"],
        "actual_roll_date": row["actual_roll_date"].strftime(
            "%Y-%m-%d"
        ),
        "deferred_new_entry_date": row[
            "deferred_new_entry_date"
        ].strftime("%Y-%m-%d"),
        "delay_days": row["delay_days"],
        "timing_effect_pnl": effect,
        "remaining_total_without_event": remaining,
        "event_share_of_total": share,
        "strategy_trade_count": row["strategy_trade_count"],
        "trade_ids": row["trade_ids"],
    })


remove_one = pd.DataFrame(remove_rows)

remove_one["absolute_effect"] = (
    remove_one["timing_effect_pnl"].abs()
)

remove_one = remove_one.sort_values(
    "absolute_effect",
    ascending=False
).drop(
    columns=["absolute_effect"]
)


# ============================================================================
# SAVE
# ============================================================================

events_out = events.copy()

for col in [
    "actual_roll_date",
    "deferred_old_last_date",
    "deferred_new_entry_date",
]:
    events_out[col] = events_out[col].dt.strftime("%Y-%m-%d")

events_out.to_csv(
    EVENT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

yearly.to_csv(
    YEARLY_FILE,
    index=False
)

delay.to_csv(
    DELAY_FILE,
    index=False
)

concentration.to_csv(
    CONCENTRATION_FILE,
    index=False
)

remove_one.to_csv(
    REMOVE_ONE_FILE,
    index=False
)

integrity.to_csv(
    INTEGRITY_FILE,
    index=False
)


# ============================================================================
# REPORT
# ============================================================================

print("\n" + "=" * 78)
print("UNIQUE MARKET EVENT RESULTS")
print("=" * 78)

print("\nDATASET")
print("-" * 78)
print(
    f"Strategy roll records       : {n_strategy_records}"
)
print(
    f"Unique market roll events   : {n_unique_events}"
)
print(
    f"Strategy/unique duplication : {duplication_factor:.2f}x"
)
print(
    f"Integrity failures           : {len(bad_integrity)}"
)

print("\nTIMING EFFECT")
print("-" * 78)
print(
    f"Total unique timing effect  : {fmt_money(total_timing)}"
)
print(
    f"Mean per unique event       : {fmt_money(mean_timing)}"
)
print(
    f"Median per unique event     : {fmt_money(median_timing)}"
)
print(
    f"Std per unique event        : {fmt_money(std_timing)}"
)

print("\nDIRECTION")
print("-" * 78)
print(
    f"Positive events             : "
    f"{positive_count} "
    f"({positive_share:.2%})"
)
print(
    f"Negative events             : "
    f"{negative_count} "
    f"({negative_count / n_unique_events:.2%})"
)
print(
    f"Neutral events              : {neutral_count}"
)

print("\nDELAY")
print("-" * 78)
print(
    f"Mean delay                  : "
    f"{events['delay_days'].mean():.2f} days"
)
print(
    f"Median delay                : "
    f"{events['delay_days'].median():.0f} days"
)
print(
    f"Maximum delay               : "
    f"{events['delay_days'].max():.0f} days"
)

print("\nYEARLY")
print("-" * 78)

print(
    yearly[
        [
            "year",
            "unique_events",
            "timing_effect_pnl",
            "mean_timing_effect_pnl",
            "positive_events",
            "negative_events",
        ]
    ].to_string(index=False)
)

print("\nDELAY BUCKETS")
print("-" * 78)

print(
    delay[
        [
            "delay_bucket",
            "unique_events",
            "timing_effect_pnl",
            "mean_timing_effect_pnl",
            "positive_events",
            "negative_events",
        ]
    ].to_string(index=False)
)

print("\nTOP UNIQUE EVENTS")
print("-" * 78)

print(
    sorted_events[
        [
            "old_contract",
            "new_contract",
            "actual_roll_date",
            "deferred_new_entry_date",
            "delay_days",
            "timing_effect_pnl",
            "strategy_trade_count",
            "trade_ids",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nBOTTOM UNIQUE EVENTS")
print("-" * 78)

print(
    sorted_events.sort_values(
        "timing_effect_pnl"
    )[
        [
            "old_contract",
            "new_contract",
            "actual_roll_date",
            "deferred_new_entry_date",
            "delay_days",
            "timing_effect_pnl",
            "strategy_trade_count",
            "trade_ids",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nCONCENTRATION")
print("-" * 78)

print(
    concentration.to_string(index=False)
)

print("\nOUTPUT FILES")
print("-" * 78)
print(EVENT_FILE)
print(SUMMARY_FILE)
print(YEARLY_FILE)
print(DELAY_FILE)
print(CONCENTRATION_FILE)
print(REMOVE_ONE_FILE)
print(INTEGRITY_FILE)

print("\n" + "=" * 78)
print("V6.4 COMPLETE")
print("=" * 78)