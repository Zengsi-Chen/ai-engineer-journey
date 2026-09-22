from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================================
# NG DEFERRED-ROLL FEASIBILITY AUDIT V6.4.1
#
# Purpose:
#   Audit deferred-roll counterfactual feasibility at BOTH:
#
#   1. roll level
#   2. trade level
#
# Important:
#   This script does NOT recalculate CME prices.
#   It uses only the V6.2 roll-attribution output.
#
# Core distinction:
#
#   A trade can be partially feasible:
#
#       Roll 1 = feasible
#       Roll 2 = infeasible
#       Roll 3 = infeasible
#
#   Therefore we must NOT classify the entire trade as infeasible.
# ============================================================================


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

INPUT_FILE = (
    DATA_DIR /
    "ng_economic_roll_attribution_v6_2_rolls.csv"
)

ROLL_OUTPUT = (
    DATA_DIR /
    "ng_deferred_feasibility_audit_v6_4_1_rolls.csv"
)

TRADE_OUTPUT = (
    DATA_DIR /
    "ng_deferred_feasibility_audit_v6_4_1_trades.csv"
)

ROLLNUM_OUTPUT = (
    DATA_DIR /
    "ng_deferred_feasibility_audit_v6_4_1_roll_number.csv"
)

EVENT_OUTPUT = (
    DATA_DIR /
    "ng_deferred_feasibility_audit_v6_4_1_events.csv"
)

SUMMARY_OUTPUT = (
    DATA_DIR /
    "ng_deferred_feasibility_audit_v6_4_1_summary.csv"
)


# ============================================================================
# REQUIRED COLUMNS
# ============================================================================

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

def money(x):
    return f"${x:,.0f}"


def pct(x):
    return f"{x:.2%}"


def normalize_date(series):
    return pd.to_datetime(
        series,
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")


def safe_join(values):
    vals = sorted(
        {
            str(v)
            for v in values
            if pd.notna(v)
        }
    )
    return ";".join(vals)


# ============================================================================
# HEADER
# ============================================================================

print("=" * 78)
print("NG DEFERRED-ROLL FEASIBILITY AUDIT V6.4.1")
print("=" * 78)


# ============================================================================
# LOAD
# ============================================================================

print("\nINPUT")
print("-" * 78)
print(f"File : {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows : {len(df)}")

missing = [
    c for c in REQUIRED_COLUMNS
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"  - {c}"
            for c in missing
        )
    )


# ============================================================================
# NORMALIZE
# ============================================================================

for col in [
    "actual_roll_date",
    "deferred_old_last_date",
    "deferred_new_entry_date",
]:
    df[col] = normalize_date(df[col])


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
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================================
# DEFINE FEASIBILITY
# ============================================================================

df["is_feasible"] = (
    df["deferred_status"]
    .eq("FULLY_RECONCILED")
)

df["is_infeasible"] = ~df["is_feasible"]


# ============================================================================
# BASIC STATUS
# ============================================================================

print("\nROLL-LEVEL STATUS")
print("-" * 78)

status = (
    df["deferred_status"]
    .value_counts(dropna=False)
)

print(status.to_string())

total_rolls = len(df)
feasible_rolls = int(df["is_feasible"].sum())
infeasible_rolls = int(df["is_infeasible"].sum())

print()
print(
    f"Total roll records       : {total_rolls}"
)
print(
    f"Feasible roll records    : {feasible_rolls}"
)
print(
    f"Infeasible roll records  : {infeasible_rolls}"
)
print(
    f"Roll-level feasibility   : "
    f"{pct(feasible_rolls / total_rolls)}"
)


# ============================================================================
# ROLL-LEVEL OUTPUT
# ============================================================================

rolls = df[
    [
        "trade_id",
        "entry_threshold",
        "exit_threshold",
        "roll_number",
        "old_contract",
        "new_contract",
        "actual_roll_date",
        "deferred_old_last_date",
        "deferred_new_entry_date",
        "delay_days",
        "deferred_status",
        "is_feasible",
    ]
].copy()

rolls.to_csv(
    ROLL_OUTPUT,
    index=False
)


# ============================================================================
# TRADE-LEVEL FEASIBILITY
# ============================================================================

trade_rows = []

for trade_id, g in df.groupby(
    "trade_id",
    sort=True
):

    total = len(g)
    feasible = int(g["is_feasible"].sum())
    infeasible = total - feasible

    if feasible == total:
        classification = "FULLY_FEASIBLE"

    elif feasible == 0:
        classification = "FULLY_INFEASIBLE"

    else:
        classification = "PARTIALLY_FEASIBLE"

    first = g.iloc[0]

    trade_rows.append({
        "trade_id": trade_id,

        "entry_threshold": first[
            "entry_threshold"
        ],

        "exit_threshold": first[
            "exit_threshold"
        ],

        "total_rolls": total,

        "feasible_rolls": feasible,

        "infeasible_rolls": infeasible,

        "feasibility_rate": (
            feasible / total
        ),

        "classification": classification,

        "feasible_roll_numbers": safe_join(
            g.loc[
                g["is_feasible"],
                "roll_number"
            ]
        ),

        "infeasible_roll_numbers": safe_join(
            g.loc[
                ~g["is_feasible"],
                "roll_number"
            ]
        ),

        "infeasible_roll_dates": safe_join(
            g.loc[
                ~g["is_feasible"],
                "actual_roll_date"
            ]
        ),

        "trade_roll_start": g[
            "actual_roll_date"
        ].min(),

        "trade_roll_end": g[
            "actual_roll_date"
        ].max(),
    })


trades = pd.DataFrame(trade_rows)

trades.to_csv(
    TRADE_OUTPUT,
    index=False
)


# ============================================================================
# TRADE-LEVEL SUMMARY
# ============================================================================

print("\nTRADE-LEVEL FEASIBILITY")
print("-" * 78)

trade_class_counts = (
    trades["classification"]
    .value_counts()
)

for name in [
    "FULLY_FEASIBLE",
    "PARTIALLY_FEASIBLE",
    "FULLY_INFEASIBLE",
]:

    count = int(
        trade_class_counts.get(
            name,
            0
        )
    )

    print(
        f"{name:25s}: {count}"
    )

print()

print(
    trades[
        [
            "trade_id",
            "total_rolls",
            "feasible_rolls",
            "infeasible_rolls",
            "feasibility_rate",
            "classification",
            "feasible_roll_numbers",
            "infeasible_roll_numbers",
        ]
    ].to_string(index=False)
)


# ============================================================================
# ROLL NUMBER FEASIBILITY
# ============================================================================

rollnum_rows = []

for roll_number, g in df.groupby(
    "roll_number",
    sort=True
):

    total = len(g)
    feasible = int(
        g["is_feasible"].sum()
    )
    infeasible = total - feasible

    rollnum_rows.append({
        "roll_number": int(roll_number),

        "total_records": total,

        "feasible_records": feasible,

        "infeasible_records": infeasible,

        "feasibility_rate": (
            feasible / total
        ),

        "infeasibility_rate": (
            infeasible / total
        ),

        "unique_trades": g[
            "trade_id"
        ].nunique(),

        "feasible_trades": g.loc[
            g["is_feasible"],
            "trade_id"
        ].nunique(),

        "infeasible_trades": g.loc[
            ~g["is_feasible"],
            "trade_id"
        ].nunique(),
    })


roll_number = pd.DataFrame(
    rollnum_rows
)

roll_number.to_csv(
    ROLLNUM_OUTPUT,
    index=False
)


# ============================================================================
# UNIQUE MARKET EVENT FEASIBILITY
#
# Same event may appear across several strategy trades.
#
# Key:
#   old_contract
#   new_contract
#   actual_roll_date
#   deferred_new_entry_date
#
# Missing deferred date means the event is infeasible.
# ============================================================================

df["unique_event_key"] = (
    df["old_contract"].astype(str)
    + "|"
    + df["new_contract"].astype(str)
    + "|"
    + df["actual_roll_date"].astype(str)
    + "|"
    + df["deferred_new_entry_date"]
        .fillna("NA")
        .astype(str)
)


event_rows = []

for key, g in df.groupby(
    "unique_event_key",
    sort=False
):

    feasible = bool(
        g["is_feasible"].any()
    )

    # If the same market event appears in multiple strategy paths,
    # it should have the same feasibility state.
    all_same = (
        g["is_feasible"].nunique() == 1
    )

    first = g.iloc[0]

    event_rows.append({
        "unique_event_key": key,

        "old_contract": first[
            "old_contract"
        ],

        "new_contract": first[
            "new_contract"
        ],

        "actual_roll_date": first[
            "actual_roll_date"
        ],

        "deferred_new_entry_date": first[
            "deferred_new_entry_date"
        ],

        "roll_number_min": g[
            "roll_number"
        ].min(),

        "roll_number_max": g[
            "roll_number"
        ].max(),

        "strategy_trade_count": len(g),

        "trade_ids": safe_join(
            g["trade_id"]
        ),

        "feasible": feasible,

        "feasibility_state_consistent": all_same,

        "status_values": safe_join(
            g["deferred_status"]
        ),
    })


events = pd.DataFrame(
    event_rows
)

events.to_csv(
    EVENT_OUTPUT,
    index=False
)


# ============================================================================
# UNIQUE EVENT SUMMARY
# ============================================================================

unique_events = len(events)

unique_feasible = int(
    events["feasible"].sum()
)

unique_infeasible = (
    unique_events
    - unique_feasible
)

unique_inconsistent = int(
    (~events["feasibility_state_consistent"])
    .sum()
)

print("\nUNIQUE MARKET EVENT FEASIBILITY")
print("-" * 78)

print(
    f"Unique market events        : "
    f"{unique_events}"
)

print(
    f"Feasible unique events      : "
    f"{unique_feasible}"
)

print(
    f"Infeasible unique events    : "
    f"{unique_infeasible}"
)

print(
    f"Unique-event feasibility    : "
    f"{pct(unique_feasible / unique_events)}"
)

print(
    f"State inconsistencies       : "
    f"{unique_inconsistent}"
)


# ============================================================================
# INFEASIBLE EVENTS
# ============================================================================

print("\nINFEASIBLE UNIQUE EVENTS")
print("-" * 78)

print(
    events[
        ~events["feasible"]
    ][
        [
            "old_contract",
            "new_contract",
            "actual_roll_date",
            "deferred_new_entry_date",
            "roll_number_min",
            "roll_number_max",
            "strategy_trade_count",
            "trade_ids",
            "status_values",
        ]
    ].to_string(index=False)
)


# ============================================================================
# ROLL-NUMBER REPORT
# ============================================================================

print("\nFEASIBILITY BY ROLL NUMBER")
print("-" * 78)

print(
    roll_number[
        [
            "roll_number",
            "total_records",
            "feasible_records",
            "infeasible_records",
            "feasibility_rate",
            "unique_trades",
        ]
    ].to_string(index=False)
)


# ============================================================================
# SUMMARY TABLE
# ============================================================================

summary_rows = [
    (
        "total_roll_records",
        total_rolls
    ),
    (
        "feasible_roll_records",
        feasible_rolls
    ),
    (
        "infeasible_roll_records",
        infeasible_rolls
    ),
    (
        "roll_level_feasibility_rate",
        feasible_rolls / total_rolls
    ),
    (
        "total_trades",
        len(trades)
    ),
    (
        "fully_feasible_trades",
        int(
            (
                trades["classification"]
                == "FULLY_FEASIBLE"
            ).sum()
        )
    ),
    (
        "partially_feasible_trades",
        int(
            (
                trades["classification"]
                == "PARTIALLY_FEASIBLE"
            ).sum()
        )
    ),
    (
        "fully_infeasible_trades",
        int(
            (
                trades["classification"]
                == "FULLY_INFEASIBLE"
            ).sum()
        )
    ),
    (
        "unique_market_events",
        unique_events
    ),
    (
        "unique_feasible_events",
        unique_feasible
    ),
    (
        "unique_infeasible_events",
        unique_infeasible
    ),
    (
        "unique_event_feasibility_rate",
        unique_feasible / unique_events
    ),
    (
        "unique_event_state_inconsistencies",
        unique_inconsistent
    ),
]


summary = pd.DataFrame(
    summary_rows,
    columns=[
        "metric",
        "value"
    ]
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================================
# OUTPUT
# ============================================================================

print("\nOUTPUT FILES")
print("-" * 78)

print(ROLL_OUTPUT)
print(TRADE_OUTPUT)
print(ROLLNUM_OUTPUT)
print(EVENT_OUTPUT)
print(SUMMARY_OUTPUT)


# ============================================================================
# FINAL
# ============================================================================

print("\n" + "=" * 78)
print("V6.4.1 COMPLETE")
print("=" * 78)