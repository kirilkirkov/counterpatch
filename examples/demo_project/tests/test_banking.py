import pytest

from banking import deposit, withdraw


def test_withdraw_reduces_balance() -> None:
    assert withdraw(100, 30) == 70


def test_withdraw_rejects_overdraft() -> None:
    with pytest.raises(ValueError):
        withdraw(10, 20)


def test_deposit_increases_balance() -> None:
    assert deposit(100, 5) == 105
