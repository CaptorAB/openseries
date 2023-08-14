"""
Defining common properties
"""
import datetime as dt
from typing import cast, TypeVar
from pandas import DataFrame

from openseries.risk import drawdown_series
from openseries.types import LiteralNanMethod

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

    def value_nan_handle(
        self: TypeCommonProps, method: LiteralNanMethod = "fill"
    ) -> TypeCommonProps:
        """Handling of missing values in a valueseries

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with last known or drop

        Returns
        -------
        self
            An object of the same class
        """
        if method == "fill":
            self.tsdf.fillna(method="pad", inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def return_nan_handle(
        self: TypeCommonProps, method: LiteralNanMethod = "fill"
    ) -> TypeCommonProps:
        """Handling of missing values in a returnseries

        Parameters
        ----------
        method: LiteralNanMethod, default: "fill"
            Method used to handle NaN. Either fill with zero or drop

        Returns
        -------
        self
            An object of the same class
        """
        if method == "fill":
            self.tsdf.fillna(value=0.0, inplace=True)
        else:
            self.tsdf.dropna(inplace=True)
        return self

    def to_drawdown_series(self: TypeCommonProps) -> TypeCommonProps:
        """Converts timeseries into a drawdown series

        Returns
        -------
        self
            An object of the same class
        """

        for serie in self.tsdf:
            self.tsdf.loc[:, serie] = drawdown_series(self.tsdf.loc[:, serie])
        return self
