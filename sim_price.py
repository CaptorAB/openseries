# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Union

from OpenSeries.stoch_processes import ModelParameters, geometric_brownian_motion_log_returns, \
    heston_model_levels, geometric_brownian_motion_jump_diffusion_levels


class ReturnSimulation(object):
    """
    A general class to hold return simulations.
    """
    number_of_sims: int
    trading_days: int
    trading_days_in_year: int
    mean_annual_return: float
    mean_annual_vol: float
    df: pd.DataFrame

    def __init__(self, d: dict):
        """
        :param d: Dictionary containing class attributes set by class method.
        """

        self.__dict__ = d

    @classmethod
    def from_normal(cls, n: int, d: int, mu: float, vol: float, t: int = 252, seed: Union[int, None] = 71):
        """
        This  function generates n number of random prices over t number of trading days.
        :param n: The number of simulations to generate.
        :param d: Number of trading days to simulate.
        :param mu: The mean return.
        :param vol: The mean standard deviation.
        :param t: The number of trading days used to annualize return and volatility.
        :param seed: This is the random seed going into numpy.random.seed(seed).
        """
        if seed:
            np.random.seed(seed)
        daily_returns = np.random.normal(loc=mu / t,
                                         scale=vol / np.sqrt(t),
                                         size=(n, d))
        output = {
            'number_of_sims': n,
            'trading_days': d,
            'trading_days_in_year': t,
            'mean_annual_return': mu,
            'mean_annual_vol': vol,
            'df': pd.DataFrame(data=daily_returns)}
        return cls(d=output)

    @classmethod
    def from_lognormal(cls, n: int, d: int, mu: float, vol: float, t: int = 252, seed: Union[int, None] = 71):
        """
        This  function generates n number of random prices over t number of trading days.
        :param n: The number of simulations to generate.
        :param d: Number of trading days to simulate.
        :param mu: The mean return.
        :param vol: The mean standard deviation.
        :param t: The number of trading days used to annualize return and volatility.
        :param seed: This is the random seed going into numpy.random.seed(seed).
        """
        if seed:
            np.random.seed(seed)
        daily_returns = np.random.lognormal(mean=mu / t,
                                            sigma=vol / np.sqrt(t),
                                            size=(n, d)) - 1
        output = {
            'number_of_sims': n,
            'trading_days': d,
            'trading_days_in_year': t,
            'mean_annual_return': mu,
            'mean_annual_vol': vol,
            'df': pd.DataFrame(data=daily_returns)}
        return cls(d=output)

    @classmethod
    def from_gbm(cls, n: int, d: int, mu: float, vol: float, t: int = 252, seed: Union[int, None] = 71):
        """
        This method constructs a sequence of log returns which, when exponentiated, produce a random Geometric Brownian
        Motion (GBM).
        :param n: The number of simulations to generate.
        :param d: Number of trading days to simulate.
        :param mu: The mean return.
        :param vol: The mean standard deviation.
        :param t: The number of trading days used to annualize return and volatility.
        :param seed: This is the random seed going into numpy.random.seed(seed).
        """
        mp = ModelParameters(all_s0=1, all_time=d, all_delta=1.0/t, all_sigma=vol, gbm_mu=mu)
        if seed:
            np.random.seed(seed)
        daily_returns = []
        for i in range(n):
            daily_returns.append(geometric_brownian_motion_log_returns(mp))
        output = {
            'number_of_sims': n,
            'trading_days': d,
            'trading_days_in_year': t,
            'mean_annual_return': mu,
            'mean_annual_vol': vol,
            'df': pd.DataFrame(data=daily_returns)}
        return cls(d=output)

    @classmethod
    def from_heston(cls, n: int, d: int, mu: float, vol: float, heston_mu: float, heston_a: float,
                    t: int = 252, seed: Union[int, None] = 71):
        """
        NOTE - this method is dodgy! Need to debug!
        The Heston model is the geometric brownian motion model with stochastic volatility.
        :param n: The number of simulations to generate.
        :param d: Number of trading days to simulate.
        :param mu: The mean return.
        :param vol: This is the volatility of the stochastic processes and the starting volatility for the Heston model.
        :param heston_mu: This is the long run average volatility for the Heston model.
        :param heston_a: This is the rate of mean reversion for volatility in the Heston model.
        :param t: The number of trading days used to annualize return and volatility.
        :param seed: This is the random seed going into numpy.random.seed(seed).
        """
        mp = ModelParameters(all_s0=1, all_time=d, all_delta=1.0/t, all_sigma=vol, gbm_mu=mu,
                             heston_vol0=vol, heston_mu=heston_mu, heston_a=heston_a)
        if seed:
            np.random.seed(seed)
        daily_returns = []
        for i in range(n):
            aray = heston_model_levels(mp)[0]
            r = aray[1:] / aray[:-1] - 1
            r = np.insert(r, 0, 0.0)
            daily_returns.append(r)
        output = {
            'number_of_sims': n,
            'trading_days': d,
            'trading_days_in_year': t,
            'mean_annual_return': mu,
            'mean_annual_vol': vol,
            'df': pd.DataFrame(data=daily_returns)}
        return cls(d=output)

    @classmethod
    def from_heston_vol(cls, n: int, d: int, mu: float, vol: float, heston_mu: float, heston_a: float,
                        t: int = 252, seed: Union[int, None] = 71):
        """

        :param n: The number of simulations to generate.
        :param d: Number of trading days to simulate.
        :param mu: The mean return.
        :param vol: This is the volatility of the stochastic processes and the starting volatility for the Heston model.
        :param heston_mu: This is the long run average volatility for the Heston model.
        :param heston_a: This is the rate of mean reversion for volatility in the Heston model.
        :param t: The number of trading days used to annualize return and volatility.
        :param seed: This is the random seed going into numpy.random.seed(seed).
        """
        mp = ModelParameters(all_s0=1, all_time=d, all_delta=1.0/t, all_sigma=vol, gbm_mu=mu,
                             heston_vol0=vol, heston_mu=heston_mu, heston_a=heston_a)
        if seed:
            np.random.seed(seed)
        daily_returns = []
        for i in range(n):
            aray = heston_model_levels(mp)[1]
            r = aray[1:] / aray[:-1] - 1
            r = np.insert(r, 0, 0.0)
            daily_returns.append(r)
        output = {
            'number_of_sims': n,
            'trading_days': d,
            'trading_days_in_year': t,
            'mean_annual_return': mu,
            'mean_annual_vol': vol,
            'df': pd.DataFrame(data=daily_returns)}
        return cls(d=output)

    @classmethod
    def from_merton_jump_gbm(cls, n: int, d: int, mu: float, vol: float, jumps_lamda: float, jumps_sigma: float,
                             jumps_mu: float, t: int = 252, seed: Union[int, None] = 71):
        """

        :param n: The number of simulations to generate.
        :param d: Number of trading days to simulate.
        :param mu: The mean return.
        :param vol: This is the volatility of the stochastic processes and the starting volatility for the Heston model.
        :param jumps_lamda: This is the probability of a jump happening at each point in time.
        :param jumps_sigma: This is the volatility of the jump size.
        :param jumps_mu: This is the average jump size.
        :param t: The number of trading days used to annualize return and volatility.
        :param seed: This is the random seed going into numpy.random.seed(seed).
        """
        mp = ModelParameters(all_s0=1, all_time=d, all_delta=1.0/t, all_sigma=vol, gbm_mu=mu,
                             jumps_lamda=jumps_lamda, jumps_sigma=jumps_sigma, jumps_mu=jumps_mu)
        if seed:
            np.random.seed(seed)
        daily_returns = []
        for i in range(n):
            aray = geometric_brownian_motion_jump_diffusion_levels(mp)
            r = aray[1:] / aray[:-1] - 1
            r = np.insert(r, 0, 0.0)
            daily_returns.append(r)
        output = {
            'number_of_sims': n,
            'trading_days': d,
            'trading_days_in_year': t,
            'mean_annual_return': mu,
            'mean_annual_vol': vol,
            'df': pd.DataFrame(data=daily_returns)}
        return cls(d=output)

    @property
    def results(self) -> pd.Series:
        return self.df.add(1.0).cumprod(axis='columns').iloc[:, -1]

    @property
    def realized_mean_return(self) -> float:
        return (self.results.mean() - 1) * self.trading_days_in_year / self.trading_days

    @property
    def realized_vol(self) -> float:
        return self.results.add(1.0).std() / np.sqrt(self.trading_days_in_year)
