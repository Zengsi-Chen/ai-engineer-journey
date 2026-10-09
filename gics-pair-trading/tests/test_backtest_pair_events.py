import pytest

from scripts.backtest_pair_events import calculate_signed_spread_return

def test_long_a_profit_when_spread_rises():
    assert calculate_signed_spread_return("LONG_A", 0.10) == pytest.approx(0.10)

def test_long_a_loss_when_spread_falls():
    assert calculate_signed_spread_return("LONG_A", -0.10) == pytest.approx(-0.10)

def test_long_b_profit_when_spread_falls():
    assert calculate_signed_spread_return("LONG_B", -0.10) == pytest.approx(0.10)

def test_long_b_loss_when_spread_rises():
    assert calculate_signed_spread_return("LONG_B", 0.10) == pytest.approx(-0.10)

def test_unknown_signal_raises_error():
    with pytest.raises(ValueError, match="Unknown signal type"):
        calculate_signed_spread_return("UNKNOWN", 0.10)
