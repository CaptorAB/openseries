import numpy as np
import pandas as pd
from typing import TypedDict

from openseries.stoch_processes import (
    ModelParameters,
    geometric_brownian_motion_log_returns,
    geometric_brownian_motion_jump_diffusion_levels,
    heston_model_levels,
)


class Simulation(TypedDict, total=False):
    """Class to hold the type of input data for the ReturnSimulation class."""

    number_of_sims: int
    trading_days: int
    trading_days_in_year: int
    mean_annual_return: float
    mean_annual_vol: float
    df: pd.DataFrame


class ReturnSimulation(object):

    number_of_sims: int
    trading_days: int
    trading_days_in_year: int
    mean_annual_return: float
    mean_annual_vol: float
    df: pd.DataFrame

    def __init__(self, d: Simulation):
        """Instantiates an object of the class ReturnSimulation

        Parameters
        ----------
        d: Simulation
            A subclass of TypedDict with the required and optional parameters
        """

        self.__dict__ = d

    @property
    def results(self) -> pd.DataFrame:
        """
        Returns
        -------
        pandas.DataFrame
            Simulation data
        """
        return self.df.add(1.0).cumprod(axis="columns").T

    @property
    def realized_mean_return(self) -> float:
        """
        Returns
        -------
        float
            Annualized arithmetic mean of returns
        """

        return float(self.results.pct_change().mean() * self.trading_days_in_year)

    @property
    def realized_vol(self) -> float:
        """
        Returns
        -------
        float
            Annualized volatility
        """

        return float(
            self.results.pct_change().std() * np.sqrt(self.trading_days_in_year)
        )

    @classmethod
    def from_normal(
        cls,
        n: int,
        d: int,
        mu: float,
        vol: float,
        t: int = 252,
        seed: int | None = 71,
    ):
        """Normal distribution simulation

        Parameters
        ----------
        n: int
            Number of simulations to generate
        d: int
            Number of trading days to simulate
        mu: float
            Mean return
        vol: float
            Mean standard deviation
        t: int, default: 252
            Number of trading days used to annualize
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Normal distribution simulation
        """

        if seed:
            np.random.seed(seed)
        daily_returns = np.random.normal(
            loc=mu / t, scale=vol / np.sqrt(t), size=(n, d)
        )
        output = Simulation(
            number_of_sims=n,
            trading_days=d,
            trading_days_in_year=t,
            mean_annual_return=mu,
            mean_annual_vol=vol,
            df=pd.DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_lognormal(
        cls,
        n: int,
        d: int,
        mu: float,
        vol: float,
        t: int = 252,
        seed: int | None = 71,
    ):
        """Lognormal distribution simulation

        Parameters
        ----------
        n: int
            Number of simulations to generate
        d: int
            Number of trading days to simulate
        mu: float
            Mean return
        vol: float
            Mean standard deviation
        t: int, default: 252
            Number of trading days used to annualize
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Lognormal distribution simulation
        """

        if seed:
            np.random.seed(seed)
        daily_returns = (
            np.random.lognormal(mean=mu / t, sigma=vol / np.sqrt(t), size=(n, d)) - 1
        )
        output = Simulation(
            number_of_sims=n,
            trading_days=d,
            trading_days_in_year=t,
            mean_annual_return=mu,
            mean_annual_vol=vol,
            df=pd.DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_gbm(
        cls,
        n: int,
        d: int,
        mu: float,
        vol: float,
        t: int = 252,
        seed: int | None = 71,
    ):
        """This method constructs a sequence of log returns which, when
        exponentiated, produce a random Geometric Brownian Motion (GBM)

        Parameters
        ----------
        n: int
            Number of simulations to generate
        d: int
            Number of trading days to simulate
        mu: float
            Mean return
        vol: float
            Mean standard deviation
        t: int, default: 252
            Number of trading days used to annualize
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Normal distribution simulation
        """

        if seed:
            np.random.seed(seed)

        mp = ModelParameters(
            all_s0=1, all_time=d, all_delta=1.0 / t, all_sigma=vol, gbm_mu=mu
        )
        daily_returns = []
        for i in range(n):
            daily_returns.append(geometric_brownian_motion_log_returns(mp))
        output = Simulation(
            number_of_sims=n,
            trading_days=d,
            trading_days_in_year=t,
            mean_annual_return=mu,
            mean_annual_vol=vol,
            df=pd.DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_heston(
        cls,
        n: int,
        d: int,
        mu: float,
        vol: float,
        heston_mu: float,
        heston_a: float,
        t: int = 252,
        seed: int | None = 71,
    ):
        """Heston model is the geometric brownian motion model with stochastic volatility

        Parameters
        ----------
        n: int
            Number of simulations to generate
        d: int
            Number of trading days to simulate
        mu: float
            Mean return
        vol: float
            Mean standard deviation
        t: int, default: 252
            Number of trading days used to annualize
        heston_mu: float
            This is the long run average volatility for the Heston model
        heston_a: float
            This is the rate of mean reversion for volatility in the Heston model
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Heston model simulation
        """

        if seed:
            np.random.seed(seed)

        mp = ModelParameters(
            all_s0=1,
            all_time=d,
            all_delta=1.0 / t,
            all_sigma=vol,
            gbm_mu=mu,
            heston_vol0=vol,
            heston_mu=heston_mu,
            heston_a=heston_a,
        )
        daily_returns = []
        for i in range(n):
            aray = heston_model_levels(mp)[0]
            r = aray[1:] / aray[:-1] - 1
            r = np.insert(r, 0, 0.0)
            daily_returns.append(r)
        output = Simulation(
            number_of_sims=n,
            trading_days=d,
            trading_days_in_year=t,
            mean_annual_return=mu,
            mean_annual_vol=vol,
            df=pd.DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_heston_vol(
        cls,
        n: int,
        d: int,
        mu: float,
        vol: float,
        heston_mu: float,
        heston_a: float,
        t: int = 252,
        seed: int | None = 71,
    ):
        """Heston Vol model simulation

        Parameters
        ----------
        n: int
            Number of simulations to generate
        d: int
            Number of trading days to simulate
        mu: float
            Mean return
        vol: float
            Mean standard deviation
        t: int, default: 252
            Number of trading days used to annualize
        heston_mu: float
            This is the long run average volatility for the Heston model
        heston_a: float
            This is the rate of mean reversion for volatility in the Heston model
        seed: int | None, default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Heston Vol model simulation
        """

        if seed:
            np.random.seed(seed)

        mp = ModelParameters(
            all_s0=1,
            all_time=d,
            all_delta=1.0 / t,
            all_sigma=vol,
            gbm_mu=mu,
            heston_vol0=vol,
            heston_mu=heston_mu,
            heston_a=heston_a,
        )
        daily_returns = []
        for i in range(n):
            aray = heston_model_levels(mp)[1]
            r = aray[1:] / aray[:-1] - 1
            r = np.insert(r, 0, 0.0)
            daily_returns.append(r)
        output = Simulation(
            number_of_sims=n,
            trading_days=d,
            trading_days_in_year=t,
            mean_annual_return=mu,
            mean_annual_vol=vol,
            df=pd.DataFrame(data=daily_returns),
        )
        return cls(d=output)

    @classmethod
    def from_merton_jump_gbm(
        cls,
        n: int,
        d: int,
        mu: float,
        vol: float,
        jumps_lamda: float,
        jumps_sigma: float,
        jumps_mu: float,
        t: int = 252,
        seed: int | None = 71,
    ):
        """Merton Jump-Diffusion model simulation

        Parameters
        ----------
        n: int
            Number of simulations to generate
        d: int
            Number of trading days to simulate
        mu: float
            Mean return
        vol: float
            Mean standard deviation
        t: int, default: 252
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
            np.random.seed(seed)

        mp = ModelParameters(
            all_s0=1,
            all_time=d,
            all_delta=1.0 / t,
            all_sigma=vol,
            gbm_mu=mu,
            jumps_lamda=jumps_lamda,
            jumps_sigma=jumps_sigma,
            jumps_mu=jumps_mu,
        )
        daily_returns = []
        for i in range(n):
            aray = geometric_brownian_motion_jump_diffusion_levels(mp)
            r = aray[1:] / aray[:-1] - 1
            r = np.insert(r, 0, 0.0)
            daily_returns.append(r)
        output = Simulation(
            number_of_sims=n,
            trading_days=d,
            trading_days_in_year=t,
            mean_annual_return=mu,
            mean_annual_vol=vol,
            df=pd.DataFrame(data=daily_returns),
        )
        return cls(d=output)
