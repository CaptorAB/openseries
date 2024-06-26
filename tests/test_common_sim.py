"""Defining simulated data used in test suite."""

import datetime as dt

from openseries.frame import OpenFrame
from openseries.series import OpenTimeSeries
from openseries.simulation import ReturnSimulation
from openseries.types import ValueType

SEED: int = 71

SIMS = ReturnSimulation.from_merton_jump_gbm(
    number_of_sims=5,
    trading_days=2512,
    mean_annual_return=0.05,
    mean_annual_vol=0.1,
    jumps_lamda=0.1,
    jumps_sigma=0.3,
    jumps_mu=-0.2,
    trading_days_in_year=252,
    seed=SEED,
)

SIMSERIES = OpenTimeSeries.from_df(
    dframe=SIMS.to_dataframe(name="Asset", end=dt.date(2019, 6, 30)),
    valuetype=ValueType.RTRN,
).to_cumret()

SIMFRAME = OpenFrame(
    [
        OpenTimeSeries.from_df(
            dframe=SIMS.to_dataframe(name="Asset", end=dt.date(2019, 6, 30)),
            column_nmbr=serie,
            valuetype=ValueType.RTRN,
        )
        for serie in range(SIMS.number_of_sims)
    ],
)
