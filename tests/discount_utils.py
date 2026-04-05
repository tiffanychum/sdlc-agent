"""
discount_utils.py
-----------------
Utility module for discount calculations.

SDLC-11: Implement discount_utils.py with pytest coverage
"""


def calculate_discount(price: float, discount_pct: float) -> float:
    """Return the price after applying a discount percentage.

    Args:
        price:        The original price (any non-negative float).
        discount_pct: The discount to apply, expressed as a percentage
                      in the closed interval [0, 100].

    Returns:
        The discounted price as a float.

    Raises:
        ValueError: If ``discount_pct`` is not in the range [0, 100].

    Examples:
        >>> calculate_discount(200, 25)
        150.0
        >>> calculate_discount(100, 0)
        100.0
        >>> calculate_discount(200, 100)
        0.0
    """
    if discount_pct < 0 or discount_pct > 100:
        raise ValueError(
            f"discount_pct must be between 0 and 100 inclusive, got {discount_pct}"
        )
    return price * (1 - discount_pct / 100)
