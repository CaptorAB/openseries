"""
Defining the ReturnSimulation class and ModelParameters used by it.

Source:
http://www.turingfinance.com/random-walks-down-wall-street-stochastic-processes-in-python/
https://github.com/StuartGordonReid/Python-Notebooks/blob/master/Stochastic%20Process%20Algorithms.ipynb

Processes that can be simulated in this module are:
- Brownian Motion
- Geometric Brownian Motion
- The Merton Jump Diffusion Model
- The Heston Stochastic Volatility Model
- Cox Ingersoll Ross
- Ornstein Uhlenbeck
"""
from __future__ import annotations

import datetime as dt
from math import log
from math import pow as mathpow
from typing import Optional, cast

from numpy import (
    add,
    append,
    array,
    exp,
    float64,
    insert,
    sqrt,
)
from numpy.random import PCG64, Generator, SeedSequence
from numpy.typing import NDArray
from pandas import DataFrame, concat
from pydantic import BaseModel, PositiveFloat, PositiveInt

from openseries.datefixer import generate_calender_date_range
from openseries.types import (
    CountriesType,
    DaysInYearType,
    ValueType,
)


def random_generator(seed: Optional[int]) -> Generator:
    """
    Make a Numpy Random Generator object.

    Parameters
    ----------
    seed: int, optional
        Random seed

    Returns
    -------
    numpy.random.Generator
        Numpy random process generator
    """
    ss = SeedSequence(entropy=seed)
    bg = PCG64(seed=ss)
    return Generator(bit_generator=bg)


def convert_to_prices(
    param: ModelParameters,
    log_returns: NDArray[float64],
) -> NDArray[float64]:
    """
    Price series.

    Converts a sequence of log returns into normal returns (exponentiation)
    and then computes a price sequence given a starting price, param.all_s0.

    Parameters
    ----------
    param: ModelParameters
        Model input
    log_returns: NDArray[float64]
        Log returns to exponentiate

    Returns
    -------
    NDArray[float64]
        Price series
    """
    returns = exp(log_returns)
    price_sequence: list[float] = [param.all_s0]
    for rtn in range(1, len(returns)):
        price_sequence.append(price_sequence[rtn - 1] * returns[rtn - 1])
    return array(price_sequence)


def brownian_motion_log_returns(
    param: ModelParameters,
    randomizer: Generator,
) -> NDArray[float64]:
    """
    Brownian Motion log returns.

    Method returns a Wiener process. The Wiener process is also called
    Brownian motion. For more information about the Wiener process check out
    the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    NDArray[float64]
        Brownian Motion log returns
    """
    sqrt_delta_sigma = sqrt(param.all_delta) * param.all_sigma
    return array(
        randomizer.normal(
            loc=0,
            scale=sqrt_delta_sigma,
            size=param.all_time,
        ),
    )


def brownian_motion_levels(
    param: ModelParameters,
    randomizer: Generator,
) -> NDArray[float64]:
    """
    Delivers a price sequence whose returns evolve as to a brownian motion.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    NDArray[float64]
        Price sequence which follows a brownian motion
    """
    return convert_to_prices(
        param=param,
        log_returns=brownian_motion_log_returns(
            param=param,
            randomizer=randomizer,
        ),
    )


def geometric_brownian_motion_log_returns(
    param: ModelParameters,
    randomizer: Generator,
) -> NDArray[float64]:
    """
    Log returns of a Geometric Brownian Motion process.

    Method constructs a sequence of log returns which, when
    exponentiated, produce a random Geometric Brownian Motion (GBM).
    GBM is the stochastic process underlying the Black Scholes
    options pricing formula.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    NDArray[float64]
        Log returns of a Geometric Brownian Motion process
    """
    wiener_process = array(
        brownian_motion_log_returns(
            param=param,
            randomizer=randomizer,
        ),
    )

    sigma_pow_mu_delta = (
        param.gbm_mu - 0.5 * mathpow(param.all_sigma, 2.0)
    ) * param.all_delta

    return wiener_process + sigma_pow_mu_delta


def geometric_brownian_motion_levels(
    param: ModelParameters,
    randomizer: Generator,
) -> NDArray[float64]:
    """
    Prices for an asset which evolves according to a geometric brownian motion.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    NDArray[float64]
        Price levels for the asset
    """
    return convert_to_prices(
        param=param,
        log_returns=geometric_brownian_motion_log_returns(
            param=param,
            randomizer=randomizer,
        ),
    )


def jump_diffusion_process(
    param: ModelParameters,
    randomizer: Generator,
) -> NDArray[float64]:
    """
    Jump sizes for each point in time (mostly zeroes if jumps are infrequent).

    Method produces a sequence of Jump Sizes which represent a jump
    diffusion process. These jumps are combined with a geometric brownian
    motion (log returns) to produce the Merton model.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    NDArray[float64]
        Jump sizes for each point in time (mostly zeroes if jumps are infrequent)
    """
    s_n = 0.0
    time = 0
    small_lamda = -(1.0 / param.jumps_lamda)
    jump_sizes = [0.0] * param.all_time

    while s_n < param.all_time:
        s_n += small_lamda * log(
            randomizer.uniform(low=0.0, high=1.0),
        )
        for j in range(param.all_time):
            if (
                time * param.all_delta
                <= s_n * param.all_delta
                <= (j + 1) * param.all_delta
            ):
                jump_sizes[j] += randomizer.normal(
                    loc=param.jumps_mu,
                    scale=param.jumps_sigma,
                )
                break
        time += 1

    return array(jump_sizes)


def geometric_brownian_motion_jump_diffusion_log_returns(
    param: ModelParameters,
    randomizer: Generator,
) -> NDArray[float64]:
    """
    Geometric Brownian Motion process with jumps in it.

    Method constructs combines a geometric brownian motion process
    (log returns) with a jump diffusion process (log returns) to produce a
    sequence of gbm jump returns.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    NDArray[float64]
        Geometric Brownian Motion process with jumps in it
    """
    jump_diffusion = jump_diffusion_process(
        param=param,
        randomizer=randomizer,
    )

    geometric_brownian_motion = geometric_brownian_motion_log_returns(
        param=param,
        randomizer=randomizer,
    )

    return add(jump_diffusion, geometric_brownian_motion)


def geometric_brownian_motion_jump_diffusion_levels(
    param: ModelParameters,
    randomizer: Generator,
) -> NDArray[float64]:
    """
    Geometric Brownian Motion generated prices.

    Converts returns generated with a Geometric Brownian Motion process
    with jumps into prices.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    NDArray[float64]
        Geometric Brownian Motion generated prices
    """
    return convert_to_prices(
        param=param,
        log_returns=geometric_brownian_motion_jump_diffusion_log_returns(
            param=param,
            randomizer=randomizer,
        ),
    )


def heston_construct_correlated_path(
    param: ModelParameters,
    brownian_motion_one: NDArray[float64],
    randomizer: Generator,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Generate correlated Brownian Motion path.

    Method is a simplified version of the Cholesky decomposition method for
    just two assets. It does not make use of matrix algebra and is therefore quite
    easy to implement.

    Parameters
    ----------
    param: ModelParameters
        Model input
    brownian_motion_one: NDArray[float64]
        A first path to correlate against
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        A correlated Brownian Motion path
    """
    sqrt_delta = sqrt(param.all_delta)

    brownian_motion_two = []
    for npath in range(param.all_time - 1):
        term_one = param.cir_rho * brownian_motion_one[npath]
        term_two = sqrt(1 - mathpow(param.cir_rho, 2.0)) * randomizer.normal(
            loc=0,
            scale=sqrt_delta,
        )
        brownian_motion_two.append(term_one + term_two)

    return array(brownian_motion_one), array(brownian_motion_two)


def cox_ingersoll_ross_heston(
    param: ModelParameters,
    randomizer: Generator,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Generate interest rate levels for the CIR process.

    Method returns the rate levels of a mean-reverting Cox Ingersoll Ross
    process. It is used to model interest rates as well as stochastic
    volatility in the Heston model. Because the returns between the underlying
    and the stochastic volatility should be correlated we pass a correlated
    Brownian motion process into the method from which the interest rate levels
    are constructed. The other correlated process is used in the Heston model.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        The interest rate levels for the CIR process
    """
    sqrt_delta_sigma = sqrt(param.all_delta) * param.all_sigma

    brownian_motion_volatility = randomizer.normal(
        loc=0,
        scale=sqrt_delta_sigma,
        size=param.all_time,
    )
    meanrev_vol, avg_vol, start_vol = (
        param.heston_a,
        param.heston_mu,
        param.heston_vol0,
    )
    volatilities: list[float] = [start_vol]
    for hpath in range(1, param.all_time):
        drift = meanrev_vol * (avg_vol - volatilities[-1]) * param.all_delta
        randomness = (
            sqrt(max(volatilities[hpath - 1], 0.05))
            * brownian_motion_volatility[hpath - 1]
        )
        volatilities.append(max(volatilities[-1], 0.05) + drift + randomness)

    return array(brownian_motion_volatility), array(volatilities)


def heston_model_levels(
    param: ModelParameters,
    randomizer: Generator,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Generate prices for an asset following a Heston process.

    The Heston model is the geometric brownian motion model with stochastic
    volatility. This stochastic volatility is given by the Cox Ingersoll Ross
    process. Step one on this method is to construct two correlated
    GBM processes. One is used for the underlying asset prices and the other
    is used for the stochastic volatility levels
    Get two correlated brownian motion sequences for the volatility parameter
    and the underlying asset brownian_motion_market,
    brownian_motion_vol = get_correlated_paths_simple(param).

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        The prices for an asset following a Heston process
    """
    brownian, cir_process = cox_ingersoll_ross_heston(
        param=param,
        randomizer=randomizer,
    )

    brownian, brownian_motion_market = heston_construct_correlated_path(
        param=param,
        brownian_motion_one=brownian,
        randomizer=randomizer,
    )

    heston_market_price_levels = array([param.all_s0])
    for hpath in range(1, param.all_time):
        drift = param.gbm_mu * heston_market_price_levels[hpath - 1] * param.all_delta
        vol = (
            cir_process[hpath - 1]
            * heston_market_price_levels[hpath - 1]
            * brownian_motion_market[hpath - 1]
        )
        heston_market_price_levels = append(
            heston_market_price_levels,
            heston_market_price_levels[hpath - 1] + drift + vol,
        )

    return array(heston_market_price_levels), array(cir_process)


def cox_ingersoll_ross_levels(
    param: ModelParameters,
    randomizer: Generator,
) -> NDArray[float64]:
    """
    Generate interest rate levels for the CIR process.

    Method returns the rate levels of a mean-reverting Cox Ingersoll Ross (CIR)
    process. It is used to model interest rates as well as stochastic
    volatility in the Heston model. Because the returns between the underlying
    and the stochastic volatility should be correlated we pass a correlated
    Brownian motion process into the method from which the interest rate levels
    are constructed. The other correlated process is used in the Heston model.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    NDArray[float64]
        The interest rate levels for the CIR process
    """
    brownian_motion = brownian_motion_log_returns(
        param=param,
        randomizer=randomizer,
    )

    levels = array([param.all_r0])
    for hpath in range(1, param.all_time):
        drift = param.cir_a * (param.cir_mu - levels[hpath - 1]) * param.all_delta
        randomness = sqrt(levels[hpath - 1]) * brownian_motion[hpath - 1]
        levels = append(levels, levels[hpath - 1] + drift + randomness)

    return array(levels)


def ornstein_uhlenbeck_levels(
    param: ModelParameters,
    randomizer: Generator,
) -> NDArray[float64]:
    """
    Generate rate levels of a mean-reverting Ornstein Uhlenbeck process.

    Parameters
    ----------
    param: ModelParameters
        Model input
    randomizer: numpy.random.Generator
        Random process generator

    Returns
    -------
    NDArray[float64]
        The interest rate levels for the Ornstein Uhlenbeck process
    """
    ou_levels = array([param.all_r0])
    brownian_motion_returns = brownian_motion_log_returns(
        param=param,
        randomizer=randomizer,
    )

    for hpath in range(1, param.all_time):
        drift = param.ou_a * (param.ou_mu - ou_levels[hpath - 1]) * param.all_delta
        randomness = brownian_motion_returns[hpath - 1]
        ou_levels = append(ou_levels, ou_levels[hpath - 1] + drift + randomness)

    return array(ou_levels)


class ReturnSimulation:

    """
    Object of the class ReturnSimulation.

    Parameters
    ----------
    number_of_sims : PositiveInt
        Number of simulations to generate
    trading_days: PositiveInt
        Total number of days to simulate
    trading_days_in_year : DaysInYearType
        Number of trading days used to annualize
    mean_annual_return : float
        Mean annual return of the distribution
    mean_annual_vol : PositiveFloat
        Mean annual standard deviation of the distribution
    dframe: pandas.DataFrame
        Pandas DataFrame object holding the resulting values
    seed: int, optional
        Seed for random process initiation
    randomizer: numpy.random.Generator, optional
        Random process generator
    """

    number_of_sims: PositiveInt
    trading_days: PositiveInt
    trading_days_in_year: DaysInYearType
    mean_annual_return: float
    mean_annual_vol: PositiveFloat
    dframe: DataFrame
    seed: Optional[int]
    randomizer: Optional[Generator]

    def __init__(
        self: ReturnSimulation,
        number_of_sims: PositiveInt,
        trading_days: PositiveInt,
        trading_days_in_year: DaysInYearType,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        dframe: DataFrame,
        seed: Optional[int] = None,
        randomizer: Optional[Generator] = None,
    ) -> None:
        """
        Object of the class ReturnSimulation.

        Parameters
        ----------
        number_of_sims : PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Total number of days to simulate
        trading_days_in_year : DaysInYearType
            Number of trading days used to annualize
        mean_annual_return : float
            Mean annual return of the distribution
        mean_annual_vol : PositiveFloat
            Mean annual standard deviation of the distribution
        dframe: pandas.DataFrame
            Pandas DataFrame object holding the resulting values
        seed: int, optional
            Seed for random process initiation
        randomizer: numpy.random.Generator, optional
            Random process generator

        Returns
        -------
        ReturnSimulation
            Object of the class ReturnSimulation
        """
        self.number_of_sims = number_of_sims
        self.trading_days = trading_days
        self.trading_days_in_year = trading_days_in_year
        self.mean_annual_return = mean_annual_return
        self.mean_annual_vol = mean_annual_vol
        self.dframe = dframe

        if randomizer:
            self.randomizer = randomizer
        else:
            self.randomizer = random_generator(seed=seed)

        self.seed = seed

    @property
    def results(self: ReturnSimulation) -> DataFrame:
        """
        Simulation data.

        Returns
        -------
        pandas.DataFrame
            Simulation data
        """
        return self.dframe.add(1.0).cumprod(axis="columns").T

    @property
    def realized_mean_return(self: ReturnSimulation) -> float:
        """
        Annualized arithmetic mean of returns.

        Returns
        -------
        float
            Annualized arithmetic mean of returns
        """
        return cast(
            float,
            (
                self.results.ffill().pct_change().mean() * self.trading_days_in_year
            ).iloc[0],
        )

    @property
    def realized_vol(self: ReturnSimulation) -> float:
        """
        Annualized volatility.

        Returns
        -------
        float
            Annualized volatility
        """
        return cast(
            float,
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
        seed: int,
        trading_days_in_year: DaysInYearType = 252,
    ) -> ReturnSimulation:
        """
        Simulate normally distributed prices.

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        seed: int
            Seed for random process initiation
        trading_days_in_year: DaysInYearType,
            default: 252
            Number of trading days used to annualize

        Returns
        -------
        ReturnSimulation
            Normally distributed prices
        """
        cls.randomizer = random_generator(seed=seed)
        daily_returns = cls.randomizer.normal(
            loc=mean_annual_return / trading_days_in_year,
            scale=mean_annual_vol / sqrt(trading_days_in_year),
            size=(number_of_sims, trading_days),
        )

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=daily_returns),
            seed=seed,
            randomizer=cls.randomizer,
        )

    @classmethod
    def from_lognormal(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        trading_days: PositiveInt,
        seed: int,
        trading_days_in_year: DaysInYearType = 252,
    ) -> ReturnSimulation:
        """
        Lognormal distribution simulation.

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        seed: int
            Seed for random process initiation
        trading_days_in_year: DaysInYearType,
            default: 252
            Number of trading days used to annualize

        Returns
        -------
        ReturnSimulation
            Lognormal distribution simulation
        """
        cls.randomizer = random_generator(seed=seed)
        daily_returns = (
            cls.randomizer.lognormal(
                mean=mean_annual_return / trading_days_in_year,
                sigma=mean_annual_vol / sqrt(trading_days_in_year),
                size=(number_of_sims, trading_days),
            )
            - 1
        )

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=daily_returns),
            seed=seed,
            randomizer=cls.randomizer,
        )

    @classmethod
    def from_gbm(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        trading_days: PositiveInt,
        seed: int,
        trading_days_in_year: DaysInYearType = 252,
    ) -> ReturnSimulation:
        """
        Geometric Brownian Motion simulation.

        Method constructs a sequence of log returns which, when
        exponentiated, produce a random Geometric Brownian Motion (GBM).

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        seed: int
            Seed for random process initiation
        trading_days_in_year: DaysInYearType,
            default: 252
            Number of trading days used to annualize

        Returns
        -------
        ReturnSimulation
            Geometric Brownian Motion simulation
        """
        model_params = ModelParameters(
            all_s0=1,
            all_time=trading_days,
            all_delta=1.0 / trading_days_in_year,
            all_sigma=mean_annual_vol,
            gbm_mu=mean_annual_return,
        )

        cls.randomizer = random_generator(seed=seed)
        daily_returns = [
            geometric_brownian_motion_log_returns(
                param=model_params,
                randomizer=cls.randomizer,
            )
            for _ in range(number_of_sims)
        ]

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=daily_returns),
            seed=seed,
            randomizer=cls.randomizer,
        )

    @classmethod
    def from_heston(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        trading_days: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        heston_mu: PositiveFloat,
        heston_a: float,
        seed: int,
        trading_days_in_year: DaysInYearType = 252,
    ) -> ReturnSimulation:
        """
        Heston model simulation.

        Heston model is the geometric brownian motion model
        with stochastic volatility.

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        heston_mu: PositiveFloat
            This is the long run average volatility for the Heston model
        heston_a: float
            This is the rate of mean reversion for volatility in the Heston model
        seed: int
            Seed for random process initiation
        trading_days_in_year: DaysInYearType,
            default: 252
            Number of trading days used to annualize

        Returns
        -------
        ReturnSimulation
            Heston model simulation
        """
        model_params = ModelParameters(
            all_s0=1,
            all_time=trading_days,
            all_delta=1.0 / trading_days_in_year,
            all_sigma=mean_annual_vol,
            gbm_mu=mean_annual_return,
            heston_vol0=mean_annual_vol,
            heston_mu=heston_mu,
            heston_a=heston_a,
        )

        cls.randomizer = random_generator(seed=seed)
        daily_returns = []
        for _ in range(number_of_sims):
            aray = heston_model_levels(
                param=model_params,
                randomizer=cls.randomizer,
            )[0]
            return_array = aray[1:] / aray[:-1] - 1
            return_array = insert(return_array, 0, 0.0)
            daily_returns.append(return_array)

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=daily_returns),
            seed=seed,
            randomizer=cls.randomizer,
        )

    @classmethod
    def from_heston_vol(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        trading_days: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        heston_mu: PositiveFloat,
        heston_a: float,
        seed: int,
        trading_days_in_year: DaysInYearType = 252,
    ) -> ReturnSimulation:
        """
        Heston Vol model simulation.

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        heston_mu: PositiveFloat
            This is the long run average volatility for the Heston model
        heston_a: float
            This is the rate of mean reversion for volatility in the Heston model
        seed: int
            Seed for random process initiation
        trading_days_in_year: DaysInYearType,
            default: 252
            Number of trading days used to annualize

        Returns
        -------
        ReturnSimulation
            Heston Vol model simulation
        """
        model_params = ModelParameters(
            all_s0=1,
            all_time=trading_days,
            all_delta=1.0 / trading_days_in_year,
            all_sigma=mean_annual_vol,
            gbm_mu=mean_annual_return,
            heston_vol0=mean_annual_vol,
            heston_mu=heston_mu,
            heston_a=heston_a,
        )

        cls.randomizer = random_generator(seed=seed)
        daily_returns = []
        for _ in range(number_of_sims):
            aray = heston_model_levels(
                param=model_params,
                randomizer=cls.randomizer,
            )[1]
            return_array = aray[1:] / aray[:-1] - 1
            return_array = insert(return_array, 0, 0.0)
            daily_returns.append(return_array)

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=daily_returns),
            seed=seed,
            randomizer=cls.randomizer,
        )

    @classmethod
    def from_merton_jump_gbm(
        cls: type[ReturnSimulation],
        number_of_sims: PositiveInt,
        trading_days: PositiveInt,
        mean_annual_return: float,
        mean_annual_vol: PositiveFloat,
        jumps_lamda: float,
        jumps_sigma: PositiveFloat,
        jumps_mu: float,
        seed: int,
        trading_days_in_year: DaysInYearType = 252,
    ) -> ReturnSimulation:
        """
        Merton Jump-Diffusion model simulation.

        Parameters
        ----------
        number_of_sims: PositiveInt
            Number of simulations to generate
        trading_days: PositiveInt
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: PositiveFloat
            Mean standard deviation
        jumps_lamda: float
            This is the probability of a jump happening at each point in time
        jumps_sigma: PositiveFloat
            This is the volatility of the jump size
        jumps_mu: float
            This is the average jump size
        seed: int
            Seed for random process initiation
        trading_days_in_year: DaysInYearType,
            default: 252
            Number of trading days used to annualize

        Returns
        -------
        ReturnSimulation
            Merton Jump-Diffusion model simulation
        """
        model_params = ModelParameters(
            all_s0=1,
            all_time=trading_days,
            all_delta=1.0 / trading_days_in_year,
            all_sigma=mean_annual_vol,
            gbm_mu=mean_annual_return,
            jumps_lamda=jumps_lamda,
            jumps_sigma=jumps_sigma,
            jumps_mu=jumps_mu,
        )

        cls.randomizer = random_generator(seed=seed)
        daily_returns = []
        for _ in range(number_of_sims):
            aray = geometric_brownian_motion_jump_diffusion_levels(
                param=model_params,
                randomizer=cls.randomizer,
            )
            return_array = aray[1:] / aray[:-1] - 1
            return_array = insert(return_array, 0, 0.0)
            daily_returns.append(return_array)

        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=daily_returns),
            seed=seed,
            randomizer=cls.randomizer,
        )

    def to_dataframe(
        self: ReturnSimulation,
        name: str,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
        countries: CountriesType = "SE",
    ) -> DataFrame:
        """
        Create pandas.DataFrame from simulation(s).

        Parameters
        ----------
        name: str
            Name label of the serie(s)
        start: datetime.date, optional
            Date when the simulation starts
        end: datetime.date, optional
            Date when the simulation ends
        countries: CountriesType, default: "SE"
            (List of) country code(s) according to ISO 3166-1 alpha-2

        Returns
        -------
        pandas.DataFrame
            Object based on the simulation(s)
        """
        d_range = generate_calender_date_range(
            trading_days=self.trading_days,
            start=start,
            end=end,
            countries=countries,
        )

        if self.number_of_sims == 1:
            sdf = self.dframe.iloc[0].T.to_frame()
            sdf.index = d_range  # type: ignore[assignment,unused-ignore]
            sdf.columns = [  # type: ignore[assignment,unused-ignore]
                [name],
                [ValueType.RTRN],
            ]
            return sdf
        fdf = DataFrame()
        for item in range(self.number_of_sims):
            sdf = self.dframe.iloc[item].T.to_frame()
            sdf.index = d_range  # type: ignore[assignment,unused-ignore]
            sdf.columns = [  # type: ignore[assignment,unused-ignore]
                [f"{name}_{item}"],
                [ValueType.RTRN],
            ]
            fdf = concat([fdf, sdf], axis="columns", sort=True)
        return fdf


class ModelParameters(BaseModel):  # type: ignore[misc, unused-ignore]

    """
    Declare ModelParameters.

    Parameters
    ----------
    all_s0: float
        Starting asset value
    all_time: PositiveInt
        Amount of time to simulate for
    all_delta: float
        Delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
    all_sigma: PositiveFloat
        Volatility of the stochastic processes
    all_r0: float, default: 0.0
        Starting interest rate value
    gbm_mu: float
        Annual drift factor for geometric brownian motion
    jumps_lamda: float, default: 0.0
        Probability of a jump happening at each point in time
    jumps_sigma: PositiveFloat, default: 0.0
        Volatility of the jump size
    jumps_mu: float, default: 0.0
        Average jump size
    cir_a: float, default: 0.0
        Rate of mean reversion for Cox Ingersoll Ross
    cir_mu: float, default: 0.0
        Long run average interest rate for Cox Ingersoll Ross
    cir_rho: float, default: 0.0
        Correlation between the wiener processes of the Heston model
    ou_a: float, default: 0.0
        Rate of mean reversion for Ornstein Uhlenbeck
    ou_mu: float, default: 0.0
        Long run average interest rate for Ornstein Uhlenbeck
    heston_a: float, default: 0.0
        Rate of mean reversion for volatility in the Heston model
    heston_mu: PositiveFloat, default: 0.0
        Long run average volatility for the Heston model
    heston_vol0: PositiveFloat, default: 0.0
        Starting volatility value for the Heston vol model
    """

    all_s0: float
    all_time: PositiveInt
    all_delta: float
    all_sigma: PositiveFloat
    gbm_mu: float
    jumps_lamda: float = 0.0
    jumps_sigma: PositiveFloat = 0.0
    jumps_mu: float = 0.0
    cir_a: float = 0.0
    cir_mu: float = 0.0
    all_r0: float = 0.0
    cir_rho: float = 0.0
    ou_a: float = 0.0
    ou_mu: float = 0.0
    heston_a: float = 0.0
    heston_mu: PositiveFloat = 0.0
    heston_vol0: PositiveFloat = 0.0
