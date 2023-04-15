"""
Simulation used in test suites
"""
import datetime as dt
from typing import Tuple
from pandas import date_range
from pandas.tseries.offsets import CustomBusinessDay

from openseries.datefixer import holiday_calendar
from openseries.frame import OpenFrame
from openseries.sim_price import ReturnSimulation
from openseries.series import (
    OpenTimeSeries,
    ValueType,
)


def make_simulated_data_from_merton_jump_gbm() -> Tuple[OpenTimeSeries, OpenFrame]:
    """Creates OpenTimeSeries and OpenFrame based on a
    Merton Jump-Diffusion model simulation

    Returns
    -------
    Tuple[OpenTimeSeries, OpenFrame]
        Objects based on a simulation
    """
    OpenTimeSeries.setup_class()

    sim = ReturnSimulation.from_merton_jump_gbm(
        number_of_sims=5,
        trading_days=2512,
        mean_annual_return=0.05,
        mean_annual_vol=0.1,
        jumps_lamda=0.00125,
        jumps_sigma=0.001,
        jumps_mu=-0.2,
        seed=71,
    )
    end = dt.date(2019, 6, 30)
    startyear = 2009
    calendar = holiday_calendar(
        startyear=startyear, endyear=end.year, countries=OpenTimeSeries.countries
    )
    d_range = [
        d.date()
        for d in date_range(
            periods=sim.trading_days,
            end=end,
            freq=CustomBusinessDay(calendar=calendar),
        )
    ]
    sdf = sim.dframe.iloc[0].T.to_frame()
    sdf.index = d_range
    sdf.columns = [["Asset"], [ValueType.RTRN]]
    series = OpenTimeSeries.from_df(sdf, valuetype=ValueType.RTRN).to_cumret()
    tslist = []
    for item in range(sim.number_of_sims):
        sdf = sim.dframe.iloc[item].T.to_frame()
        sdf.index = d_range
        sdf.columns = [[f"Asset_{item}"], [ValueType.RTRN]]
        tslist.append(OpenTimeSeries.from_df(sdf, valuetype=ValueType.RTRN))
    return series, OpenFrame(tslist)
