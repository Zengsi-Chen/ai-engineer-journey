import yfinance as yf


for ticker in ["TECK", "NRG"]:
    print("=" * 60)
    print(ticker)
    print("=" * 60)

    df = yf.Ticker(ticker).history(
        start="2026-10-07",
        end="2026-10-08",
        auto_adjust=False,
    )

    print(
        df[
            ["Open", "High", "Low", "Close", "Volume"]
        ].to_string()
    )