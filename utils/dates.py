import re
from datetime import date, datetime

# Strict but maintainable approach:
# 1) A light regex to enforce the exact ISO 8601 extended date shape YYYY-MM-DD
#    with zero-padded month/day and year 0001-9999 (Python's supported range).
# 2) Use date.fromisoformat for actual calendar validation (including leap years).
#
# Year 0000 is allowed by ISO 8601 but not by Python's datetime, so we exclude it.
_ISO_DATE_SHAPE = re.compile(
    r"^(?:000[1-9]|00[1-9]\d|0[1-9]\d{2}|[1-9]\d{3})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"
)


def parse_date(s: str) -> datetime:
    """
    Parse an ISO 8601 date string (YYYY-MM-DD).

    Returns a datetime object (at midnight) for valid strings.
    Raises ValueError for invalid format or invalid calendar dates.
    """
    if not isinstance(s, str):
        raise ValueError("Date must be a string in format YYYY-MM-DD")

    if not _ISO_DATE_SHAPE.match(s):
        raise ValueError("Invalid date format; expected YYYY-MM-DD")

    try:
        d = date.fromisoformat(s)
    except ValueError as e:
        # Covers invalid dates like 2021-02-29, 2021-04-31, etc.
        raise ValueError("Invalid calendar date") from e

    # Return a datetime at midnight to satisfy "datetime object" requirement.
    return datetime(d.year, d.month, d.day)
