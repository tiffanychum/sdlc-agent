import pytest
from datetime import datetime

from utils.dates import parse_date

# Happy path cases

def test_parse_date_valid_simple_start_of_year():
    dt = parse_date("2001-01-01")
    assert isinstance(dt, datetime) and dt == datetime(2001, 1, 1)


def test_parse_date_valid_end_of_year():
    dt = parse_date("1999-12-31")
    assert dt == datetime(1999, 12, 31)


def test_parse_date_valid_leap_day():
    dt = parse_date("2020-02-29")
    assert dt == datetime(2020, 2, 29)


# Invalid format cases (shape)

def test_parse_date_invalid_no_zero_padding_month():
    with pytest.raises(ValueError):
        parse_date("2020-2-29")


def test_parse_date_invalid_wrong_year_length():
    with pytest.raises(ValueError):
        parse_date("20-02-29")


def test_parse_date_invalid_wrong_separator():
    with pytest.raises(ValueError):
        parse_date("2020/02/29")


def test_parse_date_invalid_year_zero():
    with pytest.raises(ValueError):
        parse_date("0000-01-01")


def test_parse_date_invalid_month_13():
    with pytest.raises(ValueError):
        parse_date("2020-13-01")


def test_parse_date_invalid_month_00():
    with pytest.raises(ValueError):
        parse_date("2020-00-10")


def test_parse_date_invalid_day_00():
    with pytest.raises(ValueError):
        parse_date("2020-01-00")


def test_parse_date_invalid_day_32():
    with pytest.raises(ValueError):
        parse_date("2020-01-32")


# Invalid calendar dates (shape ok, calendar invalid)

def test_parse_date_invalid_non_leap_feb_29():
    with pytest.raises(ValueError):
        parse_date("2021-02-29")


def test_parse_date_invalid_april_31():
    with pytest.raises(ValueError):
        parse_date("2021-04-31")


def test_parse_date_invalid_november_31():
    with pytest.raises(ValueError):
        parse_date("2019-11-31")


# Non-string input

def test_parse_date_non_string_input():
    with pytest.raises(ValueError):
        parse_date(20200101)  # type: ignore[arg-type]
