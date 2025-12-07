"""Pytest fixtures for test suite."""

from __future__ import annotations

import datetime as dt
from typing import cast

import pytest

from openseries.frame import OpenFrame
from openseries.owntypes import ValueType
from openseries.series import OpenTimeSeries
from openseries.simulation import ReturnSimulation


@pytest.fixture(scope="class")
def seed() -> int:
    """Test seed value."""
    return 71


@pytest.fixture(scope="class")
def seriesim(seed: int) -> ReturnSimulation:
    """Simulated return series."""
    return ReturnSimulation.from_merton_jump_gbm(
        number_of_sims=5,
        trading_days=2512,
        mean_annual_return=0.05,
        mean_annual_vol=0.1,
        jumps_lamda=0.1,
        jumps_sigma=0.3,
        jumps_mu=-0.2,
        trading_days_in_year=252,
        seed=seed,
    )


@pytest.fixture(scope="class")
def randomseries(seriesim: ReturnSimulation) -> OpenTimeSeries:
    """Random time series for testing."""
    end_date = dt.date(2019, 6, 30)
    return OpenTimeSeries.from_df(
        dframe=seriesim.to_dataframe(name="Asset", end=end_date),
        valuetype=ValueType.RTRN,
    ).to_cumret()


@pytest.fixture(scope="class")
def randomframe(seriesim: ReturnSimulation) -> OpenFrame:
    """Random frame for testing."""
    end_date = dt.date(2019, 6, 30)
    return OpenFrame(
        constituents=[
            OpenTimeSeries.from_df(
                dframe=seriesim.to_dataframe(name="Asset", end=end_date),
                column_nmbr=serie,
                valuetype=ValueType.RTRN,
            )
            for serie in range(seriesim.number_of_sims)
        ],
    )


@pytest.fixture(scope="class")
def random_properties(
    randomseries: OpenTimeSeries,
) -> dict[str, dt.date | int | float]:
    """Random properties dictionary."""
    return cast(
        "dict[str, dt.date | int | float]",
        randomseries.all_properties().to_dict()[("Asset_0", ValueType.PRICE)],
    )


@pytest.fixture(scope="class", autouse=True)
def inject_common_fixtures(
    request: pytest.FixtureRequest,
    seed: int,
    seriesim: ReturnSimulation,
    randomseries: OpenTimeSeries,
    randomframe: OpenFrame,
    random_properties: dict[str, dt.date | int | float],
) -> None:
    """Inject common fixtures as class attributes for test classes."""
    if request.cls is not None:
        request.cls.seed = seed
        request.cls.seriesim = seriesim
        request.cls.randomseries = randomseries
        request.cls.randomframe = randomframe
        request.cls.random_properties = random_properties
