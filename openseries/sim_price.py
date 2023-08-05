"""
Defining the ReturnSimulation class which simulates returns based on
stochastic processes generated using the stoch_process.py module.
"""
import datetime as dt
from typing import cast, Optional, Tuple
from numpy import insert, random, sqrt
from pandas import DataFrame, date_range
from pandas.tseries.offsets import CustomBusinessDay
from pydantic import BaseModel

from openseries.datefixer import holiday_calendar
from openseries.frame import OpenFrame
from openseries.series import OpenTimeSeries, ValueType
from openseries.stoch_processes import (
    ModelParameters,
    geometric_brownian_motion_log_returns,
    geometric_brownian_motion_jump_diffusion_levels,
    heston_model_levels,
)


class ReturnSimulation(
    BaseModel, arbitrary_types_allowed=True, validate_assignment=True
):
    """Object of the class ReturnSimulation. Subclass of the Pydantic BaseModel

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
    dframe: pandas.DataFrame
        Pandas DataFrame object holding the resulting values
    """

    number_of_sims: int
    trading_days: int
    trading_days_in_year: int
    mean_annual_return: float
    mean_annual_vol: float
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
    def from_normal(
        cls,
        number_of_sims: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        trading_days: int,
        trading_days_in_year: int = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
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
        seed: Optional[int], default 71
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
        number_of_sims: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        trading_days: int,
        trading_days_in_year: int = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
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
        seed: Optional[int], default 71
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
        number_of_sims: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        trading_days: int,
        trading_days_in_year: int = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
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
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Normal distribution simulation
        """

        if seed:
            random.seed(seed)

        model_params = ModelParameters(
            all_s0=1,
            all_time=trading_days,
            all_delta=1.0 / trading_days_in_year,
            all_sigma=mean_annual_vol,
            gbm_mu=mean_annual_return,
        )
        daily_returns = []
        for _ in range(number_of_sims):
            daily_returns.append(geometric_brownian_motion_log_returns(model_params))
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
        number_of_sims: int,
        trading_days: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        heston_mu: float,
        heston_a: float,
        trading_days_in_year: int = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
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
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Heston model simulation
        """

        if seed:
            random.seed(seed)

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
            aray = heston_model_levels(model_params)[0]
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
        number_of_sims: int,
        trading_days: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        heston_mu: float,
        heston_a: float,
        trading_days_in_year: int = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
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
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Heston Vol model simulation
        """

        if seed:
            random.seed(seed)

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
            aray = heston_model_levels(model_params)[1]
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
        number_of_sims: int,
        trading_days: int,
        mean_annual_return: float,
        mean_annual_vol: float,
        jumps_lamda: float,
        jumps_sigma: float,
        jumps_mu: float,
        trading_days_in_year: int = 252,
        seed: Optional[int] = 71,
    ) -> "ReturnSimulation":
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
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

        Returns
        -------
        ReturnSimulation
            Merton Jump-Diffusion model simulation
        """

        if seed:
            random.seed(seed)

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
            aray = geometric_brownian_motion_jump_diffusion_levels(model_params)
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


def make_simulated_data_from_merton_jump_gbm(
    end: dt.date = dt.date(2019, 6, 30),
    startyear: int = 2009,
    number_of_sims: int = 5,
    trading_days: int = 2512,
    mean_annual_return: float = 0.05,
    mean_annual_vol: float = 0.1,
    jumps_lamda: float = 0.00125,
    jumps_sigma: float = 0.001,
    jumps_mu: float = -0.2,
    trading_days_in_year: int = 252,
    seed: Optional[int] = 71,
) -> Tuple[OpenTimeSeries, OpenFrame]:
    """Creates OpenTimeSeries and OpenFrame based on a
    Merton Jump-Diffusion model simulation

        Parameters
        ----------
        end: datetime.date, default: dt.date(2019, 6, 30)
            Date when the simulation ends
        startyear: int, default: 2009
            Year when the simulation starts. Day & month infered from end
        number_of_sims: int, default: 5
            Number of simulations to generate
        trading_days: int, default: 2512
            Number of trading days to simulate
        mean_annual_return: float, default: 0.05
            Mean return
        mean_annual_vol: float, default: 0.1
            Mean standard deviation
        trading_days_in_year: int, default: 252
            Number of trading days used to annualize
        jumps_lamda: float, default: 0.00125
            This is the probability of a jump happening at each point in time
        jumps_sigma: float, default: 0.001
            This is the volatility of the jump size
        jumps_mu: float, default: -0.2
            This is the average jump size
        seed: Optional[int], default 71
            Random seed going into numpy.random.seed()

    Returns
    -------
    Tuple[OpenTimeSeries, OpenFrame]
        Objects based on a simulation
    """
    OpenTimeSeries.setup_class()

    sim = ReturnSimulation.from_merton_jump_gbm(
        number_of_sims=number_of_sims,
        trading_days=trading_days,
        mean_annual_return=mean_annual_return,
        mean_annual_vol=mean_annual_vol,
        jumps_lamda=jumps_lamda,
        jumps_sigma=jumps_sigma,
        jumps_mu=jumps_mu,
        trading_days_in_year=trading_days_in_year,
        seed=seed,
    )
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
