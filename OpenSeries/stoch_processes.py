# -*- coding: utf-8 -*-
"""
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
import math
import numpy as np
import numpy.random as nrand
import os
from pathlib import Path
import plotly.graph_objs as go
from plotly.offline import plot
import scipy.linalg
from typing import Tuple

from OpenSeries.load_plotly import load_plotly_dict


class ModelParameters(object):
    """
    Encapsulates model parameters
    """

    def __init__(self,
                 all_s0: float, all_time: int, all_delta: float, all_sigma: float, gbm_mu: float,
                 jumps_lamda: float = 0.0, jumps_sigma: float = 0.0, jumps_mu: float = 0.0,
                 cir_a: float = 0.0, cir_mu: float = 0.0, all_r0: float = 0.0, cir_rho: float = 0.0, ou_a: float = 0.0,
                 ou_mu: float = 0.0, heston_a: float = 0.0, heston_mu: float = 0.0, heston_vol0: float = 0.0):
        """

        :param all_s0: This is the starting asset value
        :param all_time: This is the amount of time to simulate for
        :param all_delta: This is the delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
        :param all_sigma: This is the volatility of the stochastic processes
        :param all_r0: This is the starting interest rate value
        :param gbm_mu: This is the annual drift factor for geometric brownian motion
        :param jumps_lamda: This is the probability of a jump happening at each point in time
        :param jumps_sigma: This is the volatility of the jump size
        :param jumps_mu: This is the average jump size
        :param cir_a: This is the rate of mean reversion for Cox Ingersoll Ross
        :param cir_mu: This is the long run average interest rate for Cox Ingersoll Ross
        :param cir_rho: This is the correlation between the wiener processes of the Heston model
        :param ou_a: This is the rate of mean reversion for Ornstein Uhlenbeck
        :param ou_mu: This is the long run average interest rate for Ornstein Uhlenbeck
        :param heston_a: This is the rate of mean reversion for volatility in the Heston model
        :param heston_mu: This is the long run average volatility for the Heston model
        :param heston_vol0: This is the starting volatility value for the Heston model
        """
        self.all_s0 = all_s0
        self.all_time = all_time
        self.all_delta = all_delta
        self.all_sigma = all_sigma
        self.gbm_mu = gbm_mu
        self.lamda = jumps_lamda
        self.jumps_sigma = jumps_sigma
        self.jumps_mu = jumps_mu
        self.cir_a = cir_a
        self.cir_mu = cir_mu
        self.all_r0 = all_r0
        self.cir_rho = cir_rho
        self.ou_a = ou_a
        self.ou_mu = ou_mu
        self.heston_a = heston_a
        self.heston_mu = heston_mu
        self.heston_vol0 = heston_vol0


def convert_to_returns(log_returns: np.ndarray) -> np.ndarray:
    """
    This method exponentiates a sequence of log returns to get daily returns.
    :param log_returns: the log returns to exponentiated
    :return: the exponentiated returns
    """
    return np.exp(log_returns)


def convert_to_prices(param: ModelParameters, log_returns: np.ndarray) -> np.ndarray:
    """
    This method converts a sequence of log returns into normal returns (exponentiation) and then computes a price
    sequence given a starting price, param.all_s0.
    :param param: the model parameters object
    :param log_returns: the log returns to exponentiated
    :return:
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    returns = convert_to_returns(log_returns)
    # A sequence of prices starting with param.all_s0
    price_sequence: list = [param.all_s0]
    for n in range(1, len(returns)):
        # Add the price at t-1 * return at t
        price_sequence.append(price_sequence[n - 1] * returns[n - 1])
    return np.array(price_sequence)


def plot_stochastic_processes(processes: list, title: str = None) -> (go.Figure, str):
    """
    This method plots a list of stochastic processes with a specified title
    :param processes:
    :param title:
    """
    file_name = title.replace('/', '').replace('#', '').replace(' ', '').upper()
    plotfile = os.path.join(os.path.abspath(str(Path.home())), 'Documents', f'{file_name}.html')

    fig, logo = load_plotly_dict()
    figure = go.Figure(fig)

    x_axis = np.arange(0, len(processes[0]), 1)

    for n in range(len(processes)):
        figure.add_trace(go.Scatter(x=x_axis, y=processes[n],
                                    mode='lines', hovertemplate='%{y}<br>%{x}', line=dict(width=2.5, dash='solid')))

    figure.update_layout(title=dict(text=title), xaxis_title='Time, t', yaxis_title='simulated asset price',
                         showlegend=False, yaxis=dict(tickformat=None))
    figure.add_layout_image(logo)

    plot(figure, filename=plotfile, auto_open=True, link_text='', include_plotlyjs='cdn')

    return figure, plotfile


def brownian_motion_log_returns(param: ModelParameters) -> np.ndarray:
    """
    This method returns a Wiener process. The Wiener process is also called Brownian motion. For more information
    about the Wiener process check out the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process
    :param param: the model parameters object
    :return: brownian motion log returns
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    sqrt_delta_sigma = math.sqrt(param.all_delta) * param.all_sigma
    return nrand.normal(loc=0, scale=sqrt_delta_sigma, size=param.all_time)


def brownian_motion_levels(param: ModelParameters) -> np.ndarray:
    """
    Returns a price sequence whose returns evolve according to a brownian motion
    :param param: model parameters object
    :return: returns a price sequence which follows a brownian motion
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    return convert_to_prices(param, brownian_motion_log_returns(param))


def geometric_brownian_motion_log_returns(param: ModelParameters) -> np.ndarray:
    """
    This method constructs a sequence of log returns which, when exponentiated, produce a random Geometric Brownian
    Motion (GBM). GBM is the stochastic process underlying the Black Scholes options pricing formula.
    :param param: model parameters object
    :return: returns the log returns of a geometric brownian motion process
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    wiener_process = np.array(brownian_motion_log_returns(param))
    sigma_pow_mu_delta = (param.gbm_mu - 0.5 * math.pow(param.all_sigma, 2.0)) * param.all_delta
    return wiener_process + sigma_pow_mu_delta


def geometric_brownian_motion_levels(param: ModelParameters) -> np.ndarray:
    """
    Returns a sequence of price levels for an asset which evolves according to a geometric brownian motion
    :param param: model parameters object
    :return: the price levels for the asset
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    return convert_to_prices(param, geometric_brownian_motion_log_returns(param))


def jump_diffusion_process(param: ModelParameters) -> list:
    """
    This method produces a sequence of Jump Sizes which represent a jump diffusion process. These jumps are combined
    with a geometric brownian motion (log returns) to produce the Merton model.
    :param param: the model parameters object
    :return: jump sizes for each point in time (mostly zeroes if jumps are infrequent)
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    s_n = time = 0
    small_lamda = -(1.0 / param.lamda)
    jump_sizes = []
    for k in range(0, param.all_time):
        jump_sizes.append(0.0)
    while s_n < param.all_time:
        s_n += small_lamda * math.log(nrand.uniform(0, 1))
        for j in range(0, param.all_time):
            if time * param.all_delta <= s_n * param.all_delta <= (j + 1) * param.all_delta:
                # print("was true")
                jump_sizes[j] += nrand.normal(param.jumps_mu, param.jumps_sigma)
                break
        time += 1
    return jump_sizes


def geometric_brownian_motion_jump_diffusion_log_returns(param: ModelParameters) -> np.ndarray:
    """
    This method constructs combines a geometric brownian motion process (log returns) with a jump diffusion process
    (log returns) to produce a sequence of gbm jump returns.
    :param param: model parameters object
    :return: returns a GBM process with jumps in it
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    jump_diffusion = jump_diffusion_process(param)
    geometric_brownian_motion = geometric_brownian_motion_log_returns(param)
    return np.add(jump_diffusion, geometric_brownian_motion)


def geometric_brownian_motion_jump_diffusion_levels(param: ModelParameters) -> np.ndarray:
    """
    This method converts a sequence of gbm jmp returns into a price sequence which evolves according to a geometric
    brownian motion but can contain jumps at any point in time.
    :param param: model parameters object
    :return: the price levels
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    return convert_to_prices(param, geometric_brownian_motion_jump_diffusion_log_returns(param))


def heston_construct_correlated_path(param: ModelParameters,
                                     brownian_motion_one: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method is a simplified version of the Cholesky decomposition method for just two assets. It does not make use
    of matrix algebra and is therefore quite easy to implement.
    :param param: model parameters object
    :param brownian_motion_one: A first path to correlate against
    :return: a correlated brownian motion path
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    # We do not multiply by sigma here, we do that in the Heston model
    sqrt_delta = math.sqrt(param.all_delta)
    # Construct a path correlated to the first path
    brownian_motion_two = []
    for n in range(param.all_time - 1):
        term_one = param.cir_rho * brownian_motion_one[n]
        term_two = math.sqrt(1 - math.pow(param.cir_rho, 2.0)) * nrand.normal(0, sqrt_delta)
        brownian_motion_two.append(term_one + term_two)
    return np.array(brownian_motion_one), np.array(brownian_motion_two)


# noinspection PyArgumentList
def get_correlated_geometric_brownian_motions(param: ModelParameters, correlation_matrix, n: int) -> list:
    """
    This method can construct a basket of correlated asset paths using the Cholesky decomposition method
    :param param: model parameters object
    :param correlation_matrix: nxn correlation matrix
    :param n: the number of assets i.e. the number of paths to return
    :return: n correlated log return geometric brownian motion processes
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    decomposition = scipy.linalg.cholesky(correlation_matrix, lower=False)
    uncorrelated_paths = []
    sqrt_delta_sigma = math.sqrt(param.all_delta) * param.all_sigma
    # Construct uncorrelated paths to convert into correlated paths
    for h in range(param.all_time):
        uncorrelated_random_numbers = []
        for j in range(n):
            uncorrelated_random_numbers.append(nrand.normal(0, sqrt_delta_sigma))
        uncorrelated_paths.append(np.array(uncorrelated_random_numbers))
    uncorrelated_matrix = np.ndarray(uncorrelated_paths)
    correlated_matrix = uncorrelated_matrix * decomposition
    assert isinstance(correlated_matrix, np.matrix)
    # The rest of this method just extracts paths from the matrix
    extracted_paths = []
    for f in range(1, n + 1):
        extracted_paths.append([])
    for j in range(0, len(correlated_matrix)*n - n, n):
        for g in range(n):
            extracted_paths[j].append(correlated_matrix.item(j + g))
    return extracted_paths


def cox_ingersoll_ross_heston(param: ModelParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method returns the rate levels of a mean-reverting cox ingersoll ross process. It is used to model interest
    rates as well as stochastic volatility in the Heston model. Because the returns between the underlying and the
    stochastic volatility should be correlated we pass a correlated Brownian motion process into the method from which
    the interest rate levels are constructed. The other correlated process is used in the Heston model
    :param param: the model parameters objects
    :return: the interest rate levels for the CIR process
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    # We don't multiply by sigma here because we do that in heston
    sqrt_delta_sigma = math.sqrt(param.all_delta) * param.all_sigma
    brownian_motion_volatility = nrand.normal(loc=0, scale=sqrt_delta_sigma, size=param.all_time)
    a, mu, zero = param.heston_a, param.heston_mu, param.heston_vol0
    volatilities = [zero]
    for h in range(1, param.all_time):
        drift = a * (mu - volatilities[-1]) * param.all_delta
        randomness = math.sqrt(max(volatilities[h - 1], 0.05)) * brownian_motion_volatility[h-1]
        volatilities.append(max(volatilities[-1], 0.05) + drift + randomness)
    return np.array(brownian_motion_volatility), np.array(volatilities)


def heston_model_levels(param: ModelParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    NOTE - this method is dodgy! Need to debug!
    The Heston model is the geometric brownian motion model with stochastic volatility. This stochastic volatility is
    given by the cox ingersoll ross process. Step one on this method is to construct two correlated GBM processes. One
    is used for the underlying asset prices and the other is used for the stochastic volatility levels
    :param param: model parameters object
    :return: the prices for an underlying following a Heston process
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    # Get two correlated brownian motion sequences for the volatility parameter and the underlying asset
    # brownian_motion_market, brownian_motion_vol = get_correlated_paths_simple(param)
    brownian, cir_process = cox_ingersoll_ross_heston(param)
    brownian, brownian_motion_market = heston_construct_correlated_path(param, brownian)

    heston_market_price_levels = [param.all_s0]
    for h in range(1, param.all_time):
        drift = param.gbm_mu * heston_market_price_levels[h - 1] * param.all_delta
        vol = cir_process[h - 1] * heston_market_price_levels[h - 1] * brownian_motion_market[h - 1]
        heston_market_price_levels.append(heston_market_price_levels[h - 1] + drift + vol)
    return np.array(heston_market_price_levels), np.array(cir_process)


def cox_ingersoll_ross_levels(param: ModelParameters) -> np.ndarray:
    """
    This method returns the rate levels of a mean-reverting cox ingersoll ross process. It is used to model interest
    rates as well as stochastic volatility in the Heston model. Because the returns between the underlying and the
    stochastic volatility should be correlated we pass a correlated Brownian motion process into the method from which
    the interest rate levels are constructed. The other correlated process is used in the Heston model
    :param param: the model parameters object
    :return: the interest rate levels for the CIR process
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    brownian_motion = brownian_motion_log_returns(param)
    # Setup the parameters for interest rates
    a, mu, zero = param.cir_a, param.cir_mu, param.all_r0
    # Assumes output is in levels
    levels = [zero]
    for h in range(1, param.all_time):
        drift = a * (mu - levels[h-1]) * param.all_delta
        # The main difference between this and the Ornstein Uhlenbeck model is that we multiply the 'random'
        # component by the square-root of the previous level i.e. the process has level dependent interest rates.
        randomness = math.sqrt(levels[h - 1]) * brownian_motion[h - 1]
        levels.append(levels[h - 1] + drift + randomness)
    return np.array(levels)


def ornstein_uhlenbeck_levels(param: ModelParameters) -> list:
    """
    This method returns the rate levels of a mean-reverting ornstein uhlenbeck process.
    :param param: the model parameters object
    :return: the interest rate levels for the Ornstein Uhlenbeck process
    """
    assert isinstance(param, ModelParameters), 'param must be an object of Class ModelParameters'
    ou_levels = [param.all_r0]
    brownian_motion_returns = brownian_motion_log_returns(param)
    for h in range(1, param.all_time):
        drift = param.ou_a * (param.ou_mu - ou_levels[h-1]) * param.all_delta
        randomness = brownian_motion_returns[h - 1]
        ou_levels.append(ou_levels[h - 1] + drift + randomness)
    return ou_levels
