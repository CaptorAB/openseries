"""
Defining common properties
"""
import datetime as dt
from typing import cast, TypeVar
from pandas import DataFrame

TypeCommonProps = TypeVar("TypeCommonProps", bound="CommonProps")


class CommonProps:
    """Common props declared"""

    tsdf: DataFrame

    @property
    def length(self: TypeCommonProps) -> int:
        """
        Returns
        -------
        int
            Number of observations
        """

        return len(self.tsdf.index)

    @property
    def first_idx(self: TypeCommonProps) -> dt.date:
        """
        Returns
        -------
        datetime.date
            The first date in the timeseries
        """

        return cast(dt.date, self.tsdf.index[0])

    @property
    def last_idx(self: TypeCommonProps) -> dt.date:
        """
        Returns
        -------
        datetime.date
            The last date in the timeseries
        """

        return cast(dt.date, self.tsdf.index[-1])

    @property
    def span_of_days(self: TypeCommonProps) -> int:
        """
        Returns
        -------
        int
            Number of days from the first date to the last
        """

        return (self.last_idx - self.first_idx).days

    @property
    def yearfrac(self: TypeCommonProps) -> float:
        """
        Returns
        -------
        float
            Length of the timeseries expressed in years assuming all years
            have 365.25 days
        """

        return self.span_of_days / 365.25

    @property
    def periods_in_a_year(self: TypeCommonProps) -> float:
        """
        Returns
        -------
        float
            The average number of observations per year
        """

        return self.length / self.yearfrac
