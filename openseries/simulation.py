"""The ReturnSimulation class."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Self, TypedDict, cast

try:
    from typing import Unpack
except ImportError:  # pragma: no cover
    from typing_extensions import Unpack

if TYPE_CHECKING:
    import datetime as dt  # pragma: no cover

from numpy import multiply, sqrt
from numpy.random import PCG64, Generator, SeedSequence
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    concat,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

from .datefixer import generate_calendar_date_range
from .owntypes import (
    CountriesType,
    DaysInYearType,
    ValueType,
)

__all__ = ["ReturnSimulation"]


class _JumpParams(TypedDict, total=False):
    """TypedDict for jump diffusion parameters."""

    jumps_lamda: NonNegativeFloat
    jumps_sigma: NonNegativeFloat
    jumps_mu: float


def _random_generator(seed: int | None) -> Generator:
    """Make a Numpy Random Generator object.

    Args:
        seed: Random seed.

    Returns:
        Numpy random process generator.
    """
    ss = SeedSequence(entropy=seed)
    bg = PCG64(seed=cast("int | None", ss))
    return Generator(bit_generator=bg)


def _create_base_simulation(
    cls: type[ReturnSimulation],
    returns: DataFrame,
    number_of_sims: PositiveInt,
    trading_days: PositiveInt,
    trading_days_in_year: DaysInYearType,
    mean_annual_return: float,
    mean_annual_vol: PositiveFloat,
    seed: int | None = None,
    **kwargs: Unpack[_JumpParams],
) -> ReturnSimulation:
    """Common logic for creating simulations.

    Args:
        cls: The ReturnSimulation class.
        returns: The calculated returns data.
        number_of_sims: Number of simulations to generate.
        trading_days: Number of trading days to simulate.
        trading_days_in_year: Number of trading days used to annualize.
        mean_annual_return: Mean annual return.
        mean_annual_vol: Mean annual volatility.
        seed: Seed for random process initiation.
        **kwargs: Additional keyword arguments for jump parameters.

    Returns:
        A ReturnSimulation instance.
    """
    return cls(
        number_of_sims=number_of_sims,
        trading_days=trading_days,
        trading_days_in_year=trading_days_in_year,
        mean_annual_return=mean_annual_return,
        mean_annual_vol=mean_annual_vol,
        dframe=returns,
        seed=seed,
        **kwargs,
    )


class ReturnSimulation(BaseModel):
    """The class ReturnSimulation allows for simulating financial timeseries.

    Args:
        number_of_sims: Number of simulations to generate.
        trading_days: Total number of days to simulate.
        trading_days_in_year: Number of trading days used to annualize.
        mean_annual_return: Mean annual return of the distribution.
        mean_annual_vol: Mean annual standard deviation of the distribution.
        dframe: Pandas DataFrame object holding the resulting values.
        jumps_lamda: This is the probability of a jump happening at each point in time.
            Defaults to 0.0.
        jumps_sigma: This is the volatility of the jump size. Defaults to 0.0.
        jumps_mu: This is the average jump size. Defaults to 0.0.
        seed: Seed for random process initiation.

    """

    number_of_sims: PositiveInt
    trading_days: PositiveInt
    trading_days_in_year: DaysInYearType
    mean_annual_return: float
    mean_annual_vol: PositiveFloat
    dframe: DataFrame
    jumps_lamda: NonNegativeFloat = 0.0
    jumps_sigma: NonNegativeFloat = 0.0
    jumps_mu: float = 0.0
    seed: int | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="always",
    )

    @cached_property
    def results(self: Self) -> DataFrame:
        """Simulation data.

        Returns:
            Simulation data.
        """
        return self.dframe.add(1.0).cumprod(axis="columns").T

    @property
    def realized_mean_return(self: Self) -> float:
        """Annualized arithmetic mean of returns.

        Returns:
            Annualized arithmetic mean of returns.
        """
        return cast(
            "float",
            (
                self.results.ffill().pct_change().mean() * self.trading_days_in_year
            ).iloc[0],
        )

    @property
    def realized_vol(self: Self) -> float:
        """Annualized volatility.

        Returns:
            Annualized volatility.
        """
        return cast(
            "float",
            (
                self.results.ffill().pct_change().std()
                * sqrt(self.trading_days_in_year)
            ).iloc[0],
        )

    @classmethod
    def from_normal(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        trading_days: PositiveInt,
        trading_days_in_year: DaysInYearType = 252,
        seed: int | None = None,
        randomizer: Generator | None = None,
    ) -> ReturnSimulation:
        """Create a Normal distribution simulation.

        Args:
            number_of_sims: Number of simulations to generate.
            trading_days: Number of trading days to simulate.
            mean_annual_return: Mean return.
            mean_annual_vol: Mean standard deviation.
            trading_days_in_year: Number of trading days used to annualize.
                Defaults to 252.
            seed: Seed for random process initiation.
            randomizer: Random process generator.

        Returns:
            Normal distribution simulation.
        """
        if not randomizer:
            randomizer = _random_generator(seed=seed)

        returns = randomizer.normal(
            loc=mean_annual_return / trading_days_in_year,
            scale=mean_annual_vol / sqrt(trading_days_in_year),
            size=(number_of_sims, trading_days),
        )

        return _create_base_simulation(
            cls=cls,
            returns=DataFrame(data=returns, dtype="float64"),
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            seed=seed,
        )

    @classmethod
    def from_lognormal(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        trading_days: PositiveInt,
        trading_days_in_year: DaysInYearType = 252,
        seed: int | None = None,
        randomizer: Generator | None = None,
    ) -> ReturnSimulation:
        """Create a Lognormal distribution simulation.

        Args:
            number_of_sims: Number of simulations to generate.
            trading_days: Number of trading days to simulate.
            mean_annual_return: Mean return.
            mean_annual_vol: Mean standard deviation.
            trading_days_in_year: Number of trading days used to annualize.
                Defaults to 252.
            seed: Seed for random process initiation.
            randomizer: Random process generator.

        Returns:
            Lognormal distribution simulation.
        """
        if not randomizer:
            randomizer = _random_generator(seed=seed)

        returns = (
            randomizer.lognormal(
                mean=mean_annual_return / trading_days_in_year,
                sigma=mean_annual_vol / sqrt(trading_days_in_year),
                size=(number_of_sims, trading_days),
            )
            - 1
        )

        return _create_base_simulation(
            cls=cls,
            returns=DataFrame(data=returns, dtype="float64"),
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            seed=seed,
        )

    @classmethod
    def from_gbm(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        trading_days: PositiveInt,
        trading_days_in_year: DaysInYearType = 252,
        seed: int | None = None,
        randomizer: Generator | None = None,
    ) -> ReturnSimulation:
        """Create a Geometric Brownian Motion simulation.

        Args:
            number_of_sims: Number of simulations to generate.
            trading_days: Number of trading days to simulate.
            mean_annual_return: Mean return.
            mean_annual_vol: Mean standard deviation.
            trading_days_in_year: Number of trading days used to annualize.
                Defaults to 252.
            seed: Seed for random process initiation.
            randomizer: Random process generator.

        Returns:
            Geometric Brownian Motion simulation.
        """
        if not randomizer:
            randomizer = _random_generator(seed=seed)

        drift = (mean_annual_return - 0.5 * mean_annual_vol**2.0) * (
            1.0 / trading_days_in_year
        )

        normal_mean = 0.0
        wiener = randomizer.normal(
            loc=normal_mean,
            scale=sqrt(1.0 / trading_days_in_year) * mean_annual_vol,
            size=(number_of_sims, trading_days),
        )

        returns = drift + wiener

        return _create_base_simulation(
            cls=cls,
            returns=DataFrame(data=returns, dtype="float64"),
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            seed=seed,
        )

    @classmethod
    def from_merton_jump_gbm(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        trading_days: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        jumps_lamda: NonNegativeFloat,
        jumps_sigma: NonNegativeFloat = 0.0,
        jumps_mu: float = 0.0,
        trading_days_in_year: DaysInYearType = 252,
        seed: int | None = None,
        randomizer: Generator | None = None,
    ) -> ReturnSimulation:
        """Create a Merton Jump-Diffusion model simulation.

        Args:
            number_of_sims: Number of simulations to generate.
            trading_days: Number of trading days to simulate.
            mean_annual_return: Mean return.
            mean_annual_vol: Mean standard deviation.
            jumps_lamda: This is the probability of a jump happening at each point
                in time.
            jumps_sigma: This is the volatility of the jump size. Defaults to 0.0.
            jumps_mu: This is the average jump size. Defaults to 0.0.
            trading_days_in_year: Number of trading days used to annualize.
                Defaults to 252.
            seed: Seed for random process initiation.
            randomizer: Random process generator.

        Returns:
            Merton Jump-Diffusion model simulation.
        """
        if not randomizer:
            randomizer = _random_generator(seed=seed)

        normal_mean = 0.0
        wiener = randomizer.normal(
            loc=normal_mean,
            scale=sqrt(1.0 / trading_days_in_year) * mean_annual_vol,
            size=(number_of_sims, trading_days),
        )

        poisson_jumps = multiply(
            randomizer.poisson(
                lam=jumps_lamda * (1.0 / trading_days_in_year),
                size=(number_of_sims, trading_days),
            ),
            randomizer.normal(
                loc=jumps_mu,
                scale=jumps_sigma,
                size=(number_of_sims, trading_days),
            ),
        )

        drift = (
            mean_annual_return
            - 0.5 * mean_annual_vol**2.0
            - jumps_lamda * (jumps_mu + jumps_sigma**2.0)
        ) * (1.0 / trading_days_in_year)

        returns = poisson_jumps + drift + wiener

        returns[:, 0] = 0.0

        return _create_base_simulation(
            cls=cls,
            returns=DataFrame(data=returns, dtype="float64"),
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            seed=seed,
            jumps_lamda=jumps_lamda,
            jumps_sigma=jumps_sigma,
            jumps_mu=jumps_mu,
        )

    def to_dataframe(
        self: Self,
        name: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
        countries: CountriesType = "SE",
        markets: list[str] | str | None = None,
    ) -> DataFrame:
        """Create a pandas.DataFrame from simulation(s).

        Args:
            name: Name label of the serie(s).
            start: Date when the simulation starts.
            end: Date when the simulation ends.
            countries: (List of) country code(s) according to ISO 3166-1 alpha-2.
                Defaults to "SE".
            markets: (List of) markets code(s) supported by exchange_calendars.

        Returns:
            The simulation(s) data.
        """
        d_range = generate_calendar_date_range(
            trading_days=self.trading_days,
            start=start,
            end=end,
            countries=countries,
            markets=markets,
        )

        if self.number_of_sims == 1:
            sdf = self.dframe.iloc[0].T.to_frame()
            sdf.index = Index(d_range)
            sdf.columns = MultiIndex.from_arrays(
                [
                    [name],
                    [ValueType.RTRN],
                ],
            )
            return sdf

        df_list = [
            DataFrame(
                data=self.dframe.iloc[item].values,
                index=Index(d_range),
                columns=MultiIndex.from_arrays(
                    [
                        [f"{name}_{item}"],
                        [ValueType.RTRN],
                    ],
                ),
            )
            for item in range(self.number_of_sims)
        ]
        return concat(df_list, axis="columns", sort=True)
