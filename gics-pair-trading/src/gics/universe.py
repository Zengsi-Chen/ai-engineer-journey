from pathlib import Path

import yaml


CONFIG_PATH = Path("config/universe.yaml")


def load_universe():
    """Load the research universe from YAML."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def flatten_universe(universe):
    """Flatten nested universe groups into ticker records."""
    records = []

    def walk(node, path):
        if isinstance(node, list):
            for ticker in node:
                records.append(
                    {
                        "ticker": ticker,
                        "group": path[0] if len(path) > 0 else None,
                        "subcategory": path[1] if len(path) > 1 else None,
                    }
                )

        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + [key])

    walk(universe, [])

    return records


def validate_universe(records):
    """Validate ticker uniqueness."""
    tickers = [record["ticker"] for record in records]

    duplicates = sorted(
        {
            ticker
            for ticker in tickers
            if tickers.count(ticker) > 1
        }
    )

    if duplicates:
        raise ValueError(
            f"Duplicate tickers found: {duplicates}"
        )

    return True


if __name__ == "__main__":
    config = load_universe()

    records = flatten_universe(config["universe"])

    validate_universe(records)

    print(f"Universe version: {config['version']}")
    print(f"Total tickers: {len(records)}")
    print()
    print("Groups:")

    groups = {}

    for record in records:
        group = record["group"]
        groups.setdefault(group, 0)
        groups[group] += 1

    for group, count in groups.items():
        print(f"  {group}: {count}")

    print()
    print("Universe validation: PASSED")