from numpy import insert, random, sqrt
from pandas import DataFrame
from typing import TypedDict, TypeVar

from openseries.stoch_processes import (
    ModelParameters,
    geometric_brownian_motion_log_returns,
    geometric_brownian_motion_jump_diffusion_levels,
    heston_model_levels,
)

TReturnSimulation = TypeVar("TReturnSimulation", bound="ReturnSimulation")


class Simulation(TypedDict, total=False):
    """Class to hold the type of input data for the ReturnSimulation class

    Parameters
    ----------
    number_of_sims : int
        Number of simulations to generate
    trading_days: int
        Total number of days to simulate
    trading_days_in_year : str
        Number of trading days used to annualize
    mean_annual_return : List[str]
        Mean annual return of the distribution
    mean_annual_vol : str
        Mean annual standard deviation of the distribution
    df: pandas.DataFrame
        Pandas DataFrame object holding the resulting values
    """

    number_of_sims: int
    trading_days: int
    trading_days_in_year: int
    mean_annual_return: float
    mean_annual_vol: float
    df: DataFrame


class ReturnSimulation(object):
    number_of_sims: int
    trading_days: int
    trading_days_in_year: int
    mean_annual_return: float
    mean_annual_vol: float
    df: DataFrame

    def __init__(self: TReturnSimulation, d: Simulation) -> None:
        """Instantiates an object of the class ReturnSimulation

        Parameters
        ----------
        d: Simulation
            A subclass of TypedDict with the required and optional parameters
        """

        self.__dict__ = d

    @property
    def results(self: TReturnSimulation) -> DataFrame:
        """
        Returns
        -------
        pandas.DataFrame
            Simulation data
        """
        return self.df.add(1.0).cumprod(axis="columns").T

    @property
    def realized_mean_return(self: TReturnSimulation) -> float:
        """
        Returns
        -------
        float
            Annualized arithmetic mean of returns
        """

        return float(self.results.pct_change().mean() * self.trading_days_in_year)

    @property
    def realized_vol(self: TReturnSimulation) -> float:
        """
        Returns
        -------
        float
            Annualized volatility
        """

        return float(self.results.pct_change().std() * sqrt(self.trading_days_in_year))

    @classmethod
    def from_normal(
        cls,
        number_of_sims: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        trading_days: int,
        trading_days_in_year: int = 252,
        seed: int | None = 71,
    ) -> TReturnSimulation:
        """Normal distribution simulation

        Parameters
        ----------
        number_of_sims: int
            Number of simulations to generate
        trading_days: int
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: float
            Mean standard deviation
        trading_days_in_year: int, default: 252
            Number of trading days used to annualize
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Normal distribution simulation
        """

        if seed:
            random.seed(seed)
        daily_returns = random.normal(
            loc=mean_annual_return / trading_days_in_year,
            scale=mean_annual_vol / sqrt(trading_days_in_year),
            size=(number_of_sims, trading_days),
        )
        output = Simulation(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            df=DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_lognormal(
        cls,
        number_of_sims: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        trading_days: int,
        trading_days_in_year: int = 252,
        seed: int | None = 71,
    ) -> TReturnSimulation:
        """Lognormal distribution simulation

        Parameters
        ----------
        number_of_sims: int
            Number of simulations to generate
        trading_days: int
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: float
            Mean standard deviation
        trading_days_in_year: int, default: 252
            Number of trading days used to annualize
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Lognormal distribution simulation
        """

        if seed:
            random.seed(seed)
        daily_returns = (
            random.lognormal(
                mean=mean_annual_return / trading_days_in_year,
                sigma=mean_annual_vol / sqrt(trading_days_in_year),
                size=(number_of_sims, trading_days),
            )
            - 1
        )
        output = Simulation(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            df=DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_gbm(
        cls,
        number_of_sims: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        trading_days: int,
        trading_days_in_year: int = 252,
        seed: int | None = 71,
    ) -> TReturnSimulation:
        """This method constructs a sequence of log returns which, when
        exponentiated, produce a random Geometric Brownian Motion (GBM)

        Parameters
        ----------
        number_of_sims: int
            Number of simulations to generate
        trading_days: int
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: float
            Mean standard deviation
        trading_days_in_year: int, default: 252
            Number of trading days used to annualize
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Normal distribution simulation
        """

        if seed:
            random.seed(seed)

        mp = ModelParameters(
            all_s0=1,
            all_time=trading_days,
            all_delta=1.0 / trading_days_in_year,
            all_sigma=mean_annual_vol,
            gbm_mu=mean_annual_return,
        )
        daily_returns = []
        for i in range(number_of_sims):
            daily_returns.append(geometric_brownian_motion_log_returns(mp))
        output = Simulation(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            df=DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_heston(
        cls,
        number_of_sims: int,
        trading_days: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        heston_mu: float,
        heston_a: float,
        trading_days_in_year: int = 252,
        seed: int | None = 71,
    ) -> TReturnSimulation:
        """Heston model is the geometric brownian motion model
        with stochastic volatility

        Parameters
        ----------
        number_of_sims: int
            Number of simulations to generate
        trading_days: int
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: float
            Mean standard deviation
        heston_mu: float
            This is the long run average volatility for the Heston model
        heston_a: float
            This is the rate of mean reversion for volatility in the Heston model
        trading_days_in_year: int, default: 252
            Number of trading days used to annualize
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Heston model simulation
        """

        if seed:
            random.seed(seed)

        mp = ModelParameters(
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
        for i in range(number_of_sims):
            aray = heston_model_levels(mp)[0]
            r = aray[1:] / aray[:-1] - 1
            r = insert(r, 0, 0.0)
            daily_returns.append(r)
        output = Simulation(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            df=DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_heston_vol(
        cls,
        number_of_sims: int,
        trading_days: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        heston_mu: float,
        heston_a: float,
        trading_days_in_year: int = 252,
        seed: int | None = 71,
    ) -> TReturnSimulation:
        """Heston Vol model simulation

        Parameters
        ----------
        number_of_sims: int
            Number of simulations to generate
        trading_days: int
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: float
            Mean standard deviation
        heston_mu: float
            This is the long run average volatility for the Heston model
        heston_a: float
            This is the rate of mean reversion for volatility in the Heston model
        trading_days_in_year: int, default: 252
            Number of trading days used to annualize
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Heston Vol model simulation
        """

        if seed:
            random.seed(seed)

        mp = ModelParameters(
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
        for i in range(number_of_sims):
            aray = heston_model_levels(mp)[1]
            r = aray[1:] / aray[:-1] - 1
            r = insert(r, 0, 0.0)
            daily_returns.append(r)
        output = Simulation(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            df=DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_merton_jump_gbm(
        cls,
        number_of_sims: int,
        trading_days: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        jumps_lamda: float,
        jumps_sigma: float,
        jumps_mu: float,
        trading_days_in_year: int = 252,
        seed: int | None = 71,
    ) -> TReturnSimulation:
        """Merton Jump-Diffusion model simulation

        Parameters
        ----------
        number_of_sims: int
            Number of simulations to generate
        trading_days: int
            Number of trading days to simulate
        mean_annual_return: float
            Mean return
        mean_annual_vol: float
            Mean standard deviation
        trading_days_in_year: int, default: 252
            Number of trading days used to annualize
        jumps_lamda: float
            This is the probability of a jump happening at each point in time
        jumps_sigma: float
            This is the volatility of the jump size
        jumps_mu: float
            This is the average jump size
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Merton Jump-Diffusion model simulation
        """

        if seed:
            random.seed(seed)

        mp = ModelParameters(
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
        for i in range(number_of_sims):
            aray = geometric_brownian_motion_jump_diffusion_levels(mp)
            r = aray[1:] / aray[:-1] - 1
            r = insert(r, 0, 0.0)
            daily_returns.append(r)
        output = Simulation(
            number_of_sims=number_of_sims,
            trading_days=trading_days,
            trading_days_in_year=trading_days_in_year,
            mean_annual_return=mean_annual_return,
            mean_annual_vol=mean_annual_vol,
            df=DataFrame(data=daily_returns),
        )
        return cls(d=output)
