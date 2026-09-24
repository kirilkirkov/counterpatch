"""A tiny banking module used by the CounterPatch demo."""


def withdraw(balance: int, amount: int) -> int:
    if amount <= balance:
        return balance - amount

    raise ValueError("Insufficient balance")


def deposit(balance: int, amount: int) -> int:
    if amount <= 0:
        raise ValueError("Amount must be positive")

    return balance + amount
