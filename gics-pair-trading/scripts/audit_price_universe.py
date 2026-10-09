from src.gics.universe import flatten_universe, load_universe
from src.data.price_quality import validate_price_data
from src.data.yahoo_provider import YahooPriceProvider


def main():
    config = load_universe()
    records = flatten_universe(config["universe"])
    tickers = [record["ticker"] for record in records]

    provider = YahooPriceProvider()

    print("=" * 70)
    print("Yahoo Finance Price Universe Audit")
    print("=" * 70)
    print(f"Total tickers: {len(tickers)}")
    print()

    passed = []
    warnings = []
    failed = []

    for i, ticker in enumerate(tickers, start=1):
        print(
            f"[{i}/{len(tickers)}] "
            f"Testing {ticker}...",
            end=" "
        )

        try:
            df = provider.get_daily_prices(
                ticker=ticker,
                start=__import__("datetime").date(2021, 1, 1),
            )

            quality = validate_price_data(
                df,
                ticker,
            )

            status = quality["status"]

            if status == "PASSED":
                passed.append(quality)

                print(
                    f"PASS "
                    f"({quality['observations']} rows, "
                    f"{quality['start'].date()} -> "
                    f"{quality['end'].date()})"
                )

            elif status == "PASSED_WITH_WARNING":
                warnings.append(quality)

                print(
                    f"PASS WITH WARNING "
                    f"({quality['observations']} rows, "
                    f"{quality['start'].date()} -> "
                    f"{quality['end'].date()})"
                )

                for warning in quality["warnings"]:
                    print(f"  Warning: {warning}")

            else:
                failed.append(quality)

                print("FAIL")
                print(
                    f"  Errors: {quality['errors']}"
                )

        except Exception as exc:
            result = {
                "ticker": ticker,
                "errors": [str(exc)],
                "warnings": [],
            }

            failed.append(result)

            print("FAIL")
            print(f"  Error: {exc}")

    print()
    print("=" * 70)
    print("Audit Summary")
    print("=" * 70)

    print(f"Passed: {len(passed)}")
    print(f"Passed with warning: {len(warnings)}")
    print(f"Failed: {len(failed)}")
    print(f"Total: {len(tickers)}")

    if warnings:
        print()
        print("Tickers with warnings:")

        for result in warnings:
            print(
                f"  {result['ticker']}: "
                f"{result['warnings']}"
            )

    if failed:
        print()
        print("Failed tickers:")

        for result in failed:
            print(
                f"  {result['ticker']}: "
                f"{result['errors']}"
            )

        raise SystemExit(1)

    print()
    print("Price universe audit: PASSED")


if __name__ == "__main__":
    main()