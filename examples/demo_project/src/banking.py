"""A tiny banking module used by the CounterPatch demo."""


def withdraw(balance: int, amount: int) -> int:
    if amount <= 0:
        raise ValueError("Amount must be positive")

    if amount > balance:
        raise ValueError("Insufficient balance")

    return balance - amount


def deposit(balance: int, amount: int) -> int:
    if amount <= 0:
        raise ValueError("Amount must be positive")

    return balance + amount
