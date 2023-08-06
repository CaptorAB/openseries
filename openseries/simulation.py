"""
Defining the ReturnSimulation class which simulates returns based on
stochastic processes.

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
import datetime as dt
from math import log, pow as mathpow
from typing import cast, Optional, Union, Any, List, Tuple
from numpy import (
    add,
    array,
    dtype,
    exp,
    float64,
    insert,
    ndarray,
    random as nprandom,
    sqrt,
)
from pandas import DataFrame, date_range
from pandas.tseries.offsets import CustomBusinessDay
from pydantic import BaseModel, conint, confloat

from openseries.datefixer import holiday_calendar
from openseries.frame import OpenFrame
from openseries.series import OpenTimeSeries, ValueType
from openseries.types import CountriesType, CurrencyStringType


class ModelParameters(BaseModel):
    """Object of the class ModelParameters. Subclass of the Pydantic BaseModel

    Parameters
    ----------
    all_s0: float
        Starting asset value
    all_time: float
        Amount of time to simulate for
    all_delta: float
        Delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
    all_sigma: confloat(strict=True, gt=0.0)
        Volatility of the stochastic processes
    all_r0: float, default: 0.0
        Starting interest rate value
    gbm_mu: float
        Annual drift factor for geometric brownian motion
    jumps_lamda: float, default: 0.0
        Probability of a jump happening at each point in time
    jumps_sigma: confloat(ge=0.0), default: 0.0
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
    heston_mu: confloat(strict=True, gt=0.0), default: 0.0
        Long run average volatility for the Heston model
    heston_vol0: confloat(strict=True, gt=0.0), default: 0.0
        Starting volatility value for the Heston vol model
    """

    all_s0: float
    all_time: conint(strict=True, ge=1)
    all_delta: float
    all_sigma: confloat(strict=True, gt=0.0)
    gbm_mu: float
    jumps_lamda: float = 0.0
    jumps_sigma: confloat(ge=0.0) = 0.0
    jumps_mu: float = 0.0
    cir_a: float = 0.0
    cir_mu: float = 0.0
    all_r0: float = 0.0
    cir_rho: float = 0.0
    ou_a: float = 0.0
    ou_mu: float = 0.0
    heston_a: float = 0.0
    heston_mu: confloat(ge=0.0) = 0.0
    heston_vol0: confloat(ge=0.0) = 0.0


class ReturnSimulation(
    BaseModel, arbitrary_types_allowed=True, validate_assignment=True
):
    """Object of the class ReturnSimulation. Subclass of the Pydantic BaseModel

    Parameters
    ----------
    number_of_sims : int
        Number of simulations to generate
    trading_days: conint(strict=True, ge=1)
        Total number of days to simulate
    trading_days_in_year : str
        Number of trading days used to annualize
    mean_annual_return : List[str]
        Mean annual return of the distribution
    mean_annual_vol : str
        Mean annual standard deviation of the distribution
    dframe: pandas.DataFrame
        Pandas DataFrame object holding the resulting values
    """

    number_of_sims: conint(strict=True, ge=1)
    trading_days: conint(strict=True, ge=1)
    trading_days_in_year: conint(strict=True, ge=1, le=366)
    mean_annual_return: float
    mean_annual_vol: confloat(strict=True, gt=0.0)
    dframe: DataFrame

    @property
    def results(self: "ReturnSimulation") -> DataFrame:
        """
        Returns
        -------
        pandas.DataFrame
            Simulation data
        """
        return self.dframe.add(1.0).cumprod(axis="columns").T

    @property
    def realized_mean_return(self: "ReturnSimulation") -> float:
        """
        Returns
        -------
        float
            Annualized arithmetic mean of returns
        """

        return cast(
            float,
            (self.results.pct_change().mean() * self.trading_days_in_year).iloc[0],
        )

    @property
    def realized_vol(self: "ReturnSimulation") -> float:
        """
        Returns
        -------
        float
            Annualized volatility
        """

        return cast(
            float,
            (self.results.pct_change().std() * sqrt(self.trading_days_in_year)).iloc[
                0
            ],
        )

    @classmethod
    def convert_to_prices(
        cls, param: ModelParameters, log_returns: ndarray[Any, dtype[float64]]
    ) -> ndarray[Any, dtype[float64]]:
        """Converts a sequence of log returns into normal returns (exponentiation)
        and then computes a price sequence given a starting price, param.all_s0.

        Parameters
        ----------
        param: ModelParameters
            Model input
        log_returns: numpy.ndarray[Any, dtype[float64]]
            Log returns to exponentiate

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            Price series
        """

        returns = exp(log_returns)
        # A sequence of prices starting with param.all_s0
        price_sequence: List[float] = [param.all_s0]
        for rtn in range(1, len(returns)):
            # Add the price at t-1 * return at t
            price_sequence.append(price_sequence[rtn - 1] * returns[rtn - 1])
        return array(price_sequence)

    @classmethod
    def brownian_motion_log_returns(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> ndarray[Any, dtype[float64]]:
        """This method returns a Wiener process. The Wiener process is also called
        Brownian motion. For more information about the Wiener process check out
        the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            Brownian Motion log returns
        """

        if seed is not None:
            nprandom.seed(seed)

        sqrt_delta_sigma = sqrt(param.all_delta) * param.all_sigma
        return nprandom.normal(  # type: ignore[no-any-return]
            loc=0, scale=sqrt_delta_sigma, size=param.all_time
        )

    @classmethod
    def brownian_motion_levels(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> ndarray[Any, dtype[float64]]:
        """Delivers a price sequence whose returns evolve as to a brownian motion

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            Price sequence which follows a brownian motion
        """

        return cls.convert_to_prices(
            param, cls.brownian_motion_log_returns(param, seed=seed)
        )

    @classmethod
    def geometric_brownian_motion_log_returns(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> ndarray[Any, dtype[float64]]:
        """This method constructs a sequence of log returns which, when
        exponentiated, produce a random Geometric Brownian Motion (GBM).
        GBM is the stochastic process underlying the Black Scholes
        options pricing formula

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            Log returns of a Geometric Brownian Motion process
        """

        wiener_process = array(cls.brownian_motion_log_returns(param, seed=seed))
        sigma_pow_mu_delta = (
            param.gbm_mu - 0.5 * mathpow(param.all_sigma, 2.0)
        ) * param.all_delta
        return wiener_process + sigma_pow_mu_delta

    @classmethod
    def geometric_brownian_motion_levels(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> ndarray[Any, dtype[float64]]:
        """Prices for an asset which evolves according to a geometric brownian motion

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            Price levels for the asset
        """

        return cls.convert_to_prices(
            param, cls.geometric_brownian_motion_log_returns(param, seed=seed)
        )

    @classmethod
    def jump_diffusion_process(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> ndarray[Any, dtype[float64]]:
        """This method produces a sequence of Jump Sizes which represent a jump
        diffusion process. These jumps are combined with a geometric brownian
        motion (log returns) to produce the Merton model

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            Jump sizes for each point in time (mostly zeroes if jumps are infrequent)
        """

        if seed is not None:
            nprandom.seed(seed)
        s_n = 0.0
        time = 0
        small_lamda = -(1.0 / param.jumps_lamda)
        jump_sizes: List[float] = []
        for _ in range(0, param.all_time):
            jump_sizes.append(0.0)
        while s_n < param.all_time:
            s_n += small_lamda * log(nprandom.uniform(0, 1))
            for j in range(0, param.all_time):
                if (
                    time * param.all_delta
                    <= s_n * param.all_delta
                    <= (j + 1) * param.all_delta
                ):
                    jump_sizes[j] += nprandom.normal(param.jumps_mu, param.jumps_sigma)
                    break
            time += 1
        return array(jump_sizes)

    @classmethod
    def geometric_brownian_motion_jump_diffusion_log_returns(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> ndarray[Any, dtype[float64]]:
        """This method constructs combines a geometric brownian motion process
        (log returns) with a jump diffusion process (log returns) to produce a
        sequence of gbm jump returns

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            Geometric Brownian Motion process with jumps in it
        """

        jump_diffusion = cls.jump_diffusion_process(param, seed=seed)
        geometric_brownian_motion = cls.geometric_brownian_motion_log_returns(
            param, seed=seed
        )
        return add(jump_diffusion, geometric_brownian_motion)

    @classmethod
    def geometric_brownian_motion_jump_diffusion_levels(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> ndarray[Any, dtype[float64]]:
        """Converts returns generated with a Geometric Brownian Motion process
        with jumps into prices

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            Geometric Brownian Motion generated prices
        """

        return cls.convert_to_prices(
            param,
            cls.geometric_brownian_motion_jump_diffusion_log_returns(param, seed=seed),
        )

    @classmethod
    def heston_construct_correlated_path(
        cls,
        param: ModelParameters,
        brownian_motion_one: ndarray[Any, dtype[float64]],
        seed: Optional[int] = None,
    ) -> Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
        """This method is a simplified version of the Cholesky decomposition method for
        just two assets. It does not make use of matrix algebra and is therefore quite
        easy to implement

        Parameters
        ----------
        param: ModelParameters
            Model input
        brownian_motion_one: numpy.ndarray[Any, dtype[float64]]
            A first path to correlate against
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]
            A correlated Brownian Motion path
        """

        if seed is not None:
            nprandom.seed(seed)
        # We do not multiply by sigma here, we do that in the Heston model
        sqrt_delta = sqrt(param.all_delta)
        # Construct a path correlated to the first path
        brownian_motion_two = []
        for npath in range(param.all_time - 1):
            term_one = param.cir_rho * brownian_motion_one[npath]
            term_two = sqrt(1 - mathpow(param.cir_rho, 2.0)) * nprandom.normal(
                0, sqrt_delta
            )
            brownian_motion_two.append(term_one + term_two)
        return array(brownian_motion_one), array(brownian_motion_two)

    @classmethod
    def cox_ingersoll_ross_heston(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
        """This method returns the rate levels of a mean-reverting Cox Ingersoll Ross
        process. It is used to model interest rates as well as stochastic
        volatility in the Heston model. Because the returns between the underlying
        and the stochastic volatility should be correlated we pass a correlated
        Brownian motion process into the method from which the interest rate levels
        are constructed. The other correlated process is used in the Heston model

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]
            The interest rate levels for the CIR process
        """

        if seed is not None:
            nprandom.seed(seed)

        # We don't multiply by sigma here because we do that in heston
        sqrt_delta_sigma = sqrt(param.all_delta) * param.all_sigma
        brownian_motion_volatility = nprandom.normal(
            loc=0, scale=sqrt_delta_sigma, size=param.all_time
        )
        meanrev_vol, avg_vol, start_vol = (
            param.heston_a,
            param.heston_mu,
            param.heston_vol0,
        )
        volatilities: List[float] = [start_vol]
        for hpath in range(1, param.all_time):
            drift = meanrev_vol * (avg_vol - volatilities[-1]) * param.all_delta
            randomness = (
                sqrt(max(volatilities[hpath - 1], 0.05))
                * brownian_motion_volatility[hpath - 1]
            )
            volatilities.append(max(volatilities[-1], 0.05) + drift + randomness)
        return array(brownian_motion_volatility), array(volatilities)

    @classmethod
    def heston_model_levels(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
        """The Heston model is the geometric brownian motion model with stochastic
        volatility. This stochastic volatility is given by the Cox Ingersoll Ross
        process. Step one on this method is to construct two correlated
        GBM processes. One is used for the underlying asset prices and the other
        is used for the stochastic volatility levels
        Get two correlated brownian motion sequences for the volatility parameter
        and the underlying asset brownian_motion_market,
        brownian_motion_vol = get_correlated_paths_simple(param)

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]
            The prices for an asset following a Heston process
        """

        brownian, cir_process = cls.cox_ingersoll_ross_heston(param, seed=seed)
        brownian, brownian_motion_market = cls.heston_construct_correlated_path(
            param, brownian, seed=seed
        )

        heston_market_price_levels: List[float] = [param.all_s0]
        for hpath in range(1, param.all_time):
            drift = (
                param.gbm_mu * heston_market_price_levels[hpath - 1] * param.all_delta
            )
            vol = (
                cir_process[hpath - 1]
                * heston_market_price_levels[hpath - 1]
                * brownian_motion_market[hpath - 1]
            )
            heston_market_price_levels.append(
                heston_market_price_levels[hpath - 1] + drift + vol
            )
        return array(heston_market_price_levels), array(cir_process)

    @classmethod
    def cox_ingersoll_ross_levels(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> ndarray[Any, dtype[float64]]:
        """This method returns the rate levels of a mean-reverting Cox Ingersoll Ross
        process. It is used to model interest rates as well as stochastic
        volatility in the Heston model. Because the returns between the underlying
        and the stochastic volatility should be correlated we pass a correlated
        Brownian motion process into the method from which the interest rate levels
        are constructed. The other correlated process is used in the Heston model

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            The interest rate levels for the CIR process
        """

        brownian_motion = cls.brownian_motion_log_returns(param, seed=seed)

        levels: List[float] = [param.all_r0]
        for hpath in range(1, param.all_time):
            drift = param.cir_a * (param.cir_mu - levels[hpath - 1]) * param.all_delta
            randomness = sqrt(levels[hpath - 1]) * brownian_motion[hpath - 1]
            levels.append(levels[hpath - 1] + drift + randomness)
        return array(levels)

    @classmethod
    def ornstein_uhlenbeck_levels(
        cls, param: ModelParameters, seed: Optional[int] = None
    ) -> ndarray[Any, dtype[float64]]:
        """This method returns the rate levels of a mean-reverting
        Ornstein Uhlenbeck process

        Parameters
        ----------
        param: ModelParameters
            Model input
        seed: int, optional
            Random seed going into numpy.random.seed()

        Returns
        -------
        numpy.ndarray[Any, dtype[float64]]
            The interest rate levels for the Ornstein Uhlenbeck process
        """

        ou_levels: List[float] = [param.all_r0]
        brownian_motion_returns = cls.brownian_motion_log_returns(param, seed=seed)
        for hpath in range(1, param.all_time):
            drift = param.ou_a * (param.ou_mu - ou_levels[hpath - 1]) * param.all_delta
            randomness = brownian_motion_returns[hpath - 1]
            ou_levels.append(ou_levels[hpath - 1] + drift + randomness)
        return array(ou_levels)

    @classmethod
    def from_normal(
        cls,
        number_of_sims: conint(strict=True, ge=1),
        mean_annual_return: float,
        mean_annual_vol: confloat(strict=True, gt=0.0),
        trading_days: conint(strict=True, ge=1),
        trading_days_in_year: conint(strict=True, ge=1, le=366) = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
        """Normal distribution simulation

        Parameters
        ----------
        number_of_sims: conint(strict=True, ge=1)
            Number of simulations to generate
        trading_days: conint(strict=True, ge=1)
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: confloat(strict=True, gt=0.0)
            Mean standard deviation
        trading_days_in_year: conint(strict=True, ge=1, le=366), default: 252
            Number of trading days used to annualize
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Normal distribution simulation
        """

        if seed:
            nprandom.seed(seed)
        daily_returns = nprandom.normal(
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
        )

    @classmethod
    def from_lognormal(
        cls,
        number_of_sims: conint(strict=True, ge=1),
        mean_annual_return: float,
        mean_annual_vol: confloat(strict=True, gt=0.0),
        trading_days: conint(strict=True, ge=1),
        trading_days_in_year: conint(strict=True, ge=1, le=366) = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
        """Lognormal distribution simulation

        Parameters
        ----------
        number_of_sims: conint(strict=True, ge=1)
            Number of simulations to generate
        trading_days: conint(strict=True, ge=1)
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: confloat(strict=True, gt=0.0)
            Mean standard deviation
        trading_days_in_year: conint(strict=True, ge=1, le=366), default: 252
            Number of trading days used to annualize
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Lognormal distribution simulation
        """

        if seed:
            nprandom.seed(seed)
        daily_returns = (
            nprandom.lognormal(
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
        )

    @classmethod
    def from_gbm(
        cls,
        number_of_sims: conint(strict=True, ge=1),
        mean_annual_return: float,
        mean_annual_vol: confloat(strict=True, gt=0.0),
        trading_days: conint(strict=True, ge=1),
        trading_days_in_year: conint(strict=True, ge=1, le=366) = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
        """This method constructs a sequence of log returns which, when
        exponentiated, produce a random Geometric Brownian Motion (GBM)

        Parameters
        ----------
        number_of_sims: conint(strict=True, ge=1)
            Number of simulations to generate
        trading_days: conint(strict=True, ge=1)
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: confloat(strict=True, gt=0.0)
            Mean standard deviation
        trading_days_in_year: conint(strict=True, ge=1, le=366), default: 252
            Number of trading days used to annualize
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Normal distribution simulation
        """

        if seed:
            nprandom.seed(seed)

        model_params = ModelParameters(
            all_s0=1,
            all_time=trading_days,
            all_delta=1.0 / trading_days_in_year,
            all_sigma=mean_annual_vol,
            gbm_mu=mean_annual_return,
        )
        daily_returns = []
        for _ in range(number_of_sims):
            daily_returns.append(
                cls.geometric_brownian_motion_log_returns(param=model_params)
            )
        return cls(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            dframe=DataFrame(data=daily_returns),
        )

    @classmethod
    def from_heston(
        cls,
        number_of_sims: conint(strict=True, ge=1),
        trading_days: conint(strict=True, ge=1),
        mean_annual_return: float,
        mean_annual_vol: confloat(strict=True, gt=0.0),
        heston_mu: float,
        heston_a: float,
        trading_days_in_year: conint(strict=True, ge=1, le=366) = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
        """Heston model is the geometric brownian motion model
        with stochastic volatility

        Parameters
        ----------
        number_of_sims: conint(strict=True, ge=1)
            Number of simulations to generate
        trading_days: conint(strict=True, ge=1)
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: confloat(strict=True, gt=0.0)
            Mean standard deviation
        heston_mu: float
            This is the long run average volatility for the Heston model
        heston_a: float
            This is the rate of mean reversion for volatility in the Heston model
        trading_days_in_year: conint(strict=True, ge=1, le=366), default: 252
            Number of trading days used to annualize
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Heston model simulation
        """

        if seed:
            nprandom.seed(seed)

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
        daily_returns = []
        for _ in range(number_of_sims):
            aray = cls.heston_model_levels(model_params)[0]
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
        )

    @classmethod
    def from_heston_vol(
        cls,
        number_of_sims: conint(strict=True, ge=1),
        trading_days: conint(strict=True, ge=1),
        mean_annual_return: float,
        mean_annual_vol: confloat(strict=True, gt=0.0),
        heston_mu: float,
        heston_a: float,
        trading_days_in_year: conint(strict=True, ge=1, le=366) = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
        """Heston Vol model simulation

        Parameters
        ----------
        number_of_sims: conint(strict=True, ge=1)
            Number of simulations to generate
        trading_days: conint(strict=True, ge=1)
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: confloat(strict=True, gt=0.0)
            Mean standard deviation
        heston_mu: float
            This is the long run average volatility for the Heston model
        heston_a: float
            This is the rate of mean reversion for volatility in the Heston model
        trading_days_in_year: conint(strict=True, ge=1, le=366), default: 252
            Number of trading days used to annualize
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Heston Vol model simulation
        """

        if seed:
            nprandom.seed(seed)

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
        daily_returns = []
        for _ in range(number_of_sims):
            aray = cls.heston_model_levels(model_params)[1]
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
        )

    @classmethod
    def from_merton_jump_gbm(
        cls,
        number_of_sims: conint(strict=True, ge=1),
        trading_days: conint(strict=True, ge=1),
        mean_annual_return: float,
        mean_annual_vol: confloat(strict=True, gt=0.0),
        jumps_lamda: float,
        jumps_sigma: float,
        jumps_mu: float,
        trading_days_in_year: conint(strict=True, ge=1, le=366) = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
        """Merton Jump-Diffusion model simulation

        Parameters
        ----------
        number_of_sims: conint(strict=True, ge=1)
            Number of simulations to generate
        trading_days: conint(strict=True, ge=1)
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: confloat(strict=True, gt=0.0)
            Mean standard deviation
        trading_days_in_year: conint(strict=True, ge=1, le=366), default: 252
            Number of trading days used to annualize
        jumps_lamda: float
            This is the probability of a jump happening at each point in time
        jumps_sigma: float
            This is the volatility of the jump size
        jumps_mu: float
            This is the average jump size
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Merton Jump-Diffusion model simulation
        """

        if seed:
            nprandom.seed(seed)

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
        daily_returns = []
        for _ in range(number_of_sims):
            aray = cls.geometric_brownian_motion_jump_diffusion_levels(model_params)
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
        )

    def to_opentimeseries_openframe(
        self: "ReturnSimulation",
        name: str,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
        valuetype: Union[ValueType.PRICE, ValueType.RTRN] = ValueType.PRICE,
        baseccy: CurrencyStringType = "SEK",
        countries: CountriesType = "SE",
    ) -> Union[OpenTimeSeries, OpenFrame]:
        """Creates OpenTimeSeries or OpenFrame from the simulation(s)

            Parameters
            ----------
            name: str
                Name label of the serie(s)
            start: datetime.date, optional
                Date when the simulation starts
            end: datetime.date, optional
                Date when the simulation ends
            valuetype : Union[ValueType.PRICE, ValueType.RTRN, default: ValueType.PRICE
                Identifies if the series is a series of values or returns
            baseccy : str, default: "SEK"
                ISO 4217 currency code of the timeseries
            countries: Union[List[str], str], default: "SE"
                (List of) country code(s) according to ISO 3166-1 alpha-2

        Returns
        -------
        Union[OpenTimeSeries, OpenFrame]
            Object based on the simulation(s)
        """
        if start and not end:
            tmp_range = date_range(
                start=start, periods=self.trading_days * 365 // 252, freq="D"
            )
            calendar = holiday_calendar(
                startyear=start.year, endyear=tmp_range[-1].year, countries=countries
            )
            d_range = [
                d.date()
                for d in date_range(
                    start=start,
                    periods=self.trading_days,
                    freq=CustomBusinessDay(calendar=calendar),
                )
            ]

        elif end and not start:
            tmp_range = date_range(
                end=end, periods=self.trading_days * 365 // 252, freq="D"
            )
            calendar = holiday_calendar(
                startyear=tmp_range[0].year, endyear=end.year, countries=countries
            )
            d_range = [
                d.date()
                for d in date_range(
                    end=end,
                    periods=self.trading_days,
                    freq=CustomBusinessDay(calendar=calendar),
                )
            ]
        else:
            raise ValueError(
                "Provide one of start or end date, but not both. "
                "Date range is inferred from number of trading days."
            )

        if self.number_of_sims == 1:
            sdf = self.dframe.iloc[0].T.to_frame()
            sdf.index = d_range
            sdf.columns = [[name], [ValueType.RTRN]]
            if valuetype == ValueType.PRICE:
                return OpenTimeSeries.from_df(
                    dframe=sdf,
                    valuetype=ValueType.RTRN,
                    baseccy=baseccy,
                ).to_cumret()
            return OpenTimeSeries.from_df(
                dframe=sdf,
                valuetype=ValueType.RTRN,
                baseccy=baseccy,
            )
        tslist = []
        for item in range(self.number_of_sims):
            sdf = self.dframe.iloc[item].T.to_frame()
            sdf.index = d_range
            sdf.columns = [[f"Asset_{item}"], [ValueType.RTRN]]
            if valuetype == ValueType.PRICE:
                tslist.append(
                    OpenTimeSeries.from_df(sdf, valuetype=ValueType.RTRN).to_cumret()
                )
            else:
                tslist.append(OpenTimeSeries.from_df(sdf, valuetype=ValueType.RTRN))
        return OpenFrame(tslist)
