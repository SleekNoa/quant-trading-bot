def calculate_position_size(balance: float, risk_percent: float, price: float) -> int:
    """
    Returns the number of whole shares to buy based on
    a fixed percentage of the current account balance.

    Example:
        balance      = $10,000
        risk_percent = 0.01  (1%)
        price        = $150
        → risk_amount = $100
        → shares      = 0  (floor of 100/150)

    For demo purposes we also expose a 'full_position' helper that
    simply buys as many shares as cash allows.
    """
    risk_amount = balance * risk_percent
    shares = int(risk_amount / price)
    return shares


def full_position_size(balance: float, price: float) -> int:
    """Buy as many shares as the balance allows (used by backtester)."""
    return int(balance / price)
