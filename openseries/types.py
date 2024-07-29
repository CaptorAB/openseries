"""Declaring types used throughout the project."""
from enum import Enum

__all__ = ["ValueType"]


class ValueType(str, Enum):
    """Enum types of OpenTimeSeries to identify the output."""

    EWMA = "EWMA"
    PRICE = "Price(Close)"
    RTRN = "Return(Total)"
    RELRTRN = "Relative return"
    ROLLBETA = "Beta"
    ROLLCORR = "Rolling correlation"
    ROLLCVAR = "Rolling CVaR"
    ROLLINFORATIO = "Information Ratio"
    ROLLRTRN = "Rolling returns"
    ROLLVAR = "Rolling VaR"
    ROLLVOL = "Rolling volatility"
