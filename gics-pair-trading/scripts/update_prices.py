from src.gics.universe import flatten_universe, load_universe
from src.data.price_updater import update_ticker


def main():
    config = load_universe()

    records = flatten_universe(
        config["universe"]
    )

    tickers = [
        record["ticker"]
        for record in records
    ]

    print("=" * 60)
    print("GICS Pair Trading - Price Update")
    print("=" * 60)
    print(f"Total tickers: {len(tickers)}")
    print()

    results = []

    for i, ticker in enumerate(tickers, start=1):

        print(
            f"[{i}/{len(tickers)}] "
            f"Updating {ticker}..."
        )

        try:
            result = update_ticker(ticker)

            results.append(result)

            print(
                f"  Mode: {result['mode']}"
            )

            print(
                f"  Rows added: "
                f"{result['rows_added']}"
            )

            print(
                f"  Total rows: "
                f"{result['total_rows']}"
            )

            print(
                f"  Last date: "
                f"{result['last_date']}"
            )

            print(
                f"  Quality: "
                f"{result['quality']['status']}"
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            results.append(
                {
                    "ticker": ticker,
                    "mode": "failed",
                    "error": str(exc),
                }
            )

        print()

    print("=" * 60)
    print("Update Summary")
    print("=" * 60)

    modes = {}

    for result in results:

        mode = result["mode"]

        modes.setdefault(
            mode,
            0,
        )

        modes[mode] += 1

    for mode, count in modes.items():

        print(
            f"{mode}: {count}"
        )

    print()

    failed = [
        result
        for result in results
        if result["mode"] == "failed"
    ]

    if failed:

        print("Failed tickers:")

        for result in failed:

            print(
                f"  {result['ticker']}: "
                f"{result['error']}"
            )

        raise SystemExit(1)

    print("Price update: PASSED")


if __name__ == "__main__":
    main()