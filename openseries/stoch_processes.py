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


class ModelParameters(object):
    def __init__(
        self,
        all_s0: float,
        all_time: int,
        all_delta: float,
        all_sigma: float,
        gbm_mu: float,
        jumps_lamda: float = 0.0,
        jumps_sigma: float = 0.0,
        jumps_mu: float = 0.0,
        cir_a: float = 0.0,
        cir_mu: float = 0.0,
        all_r0: float = 0.0,
        cir_rho: float = 0.0,
        ou_a: float = 0.0,
        ou_mu: float = 0.0,
        heston_a: float = 0.0,
        heston_mu: float = 0.0,
        heston_vol0: float = 0.0,
    ):
        """Instantiates an object of the class ModelParameters

        Parameters
        ----------
        all_s0: float
            Starting asset value
        all_time: float
            Amount of time to simulate for
        all_delta: float
            Delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
        all_sigma: float
            Volatility of the stochastic processes
        all_r0: float, default: 0.0
            Starting interest rate value
        gbm_mu: float
            Annual drift factor for geometric brownian motion
        jumps_lamda: float, default: 0.0
            Probability of a jump happening at each point in time
        jumps_sigma: float, default: 0.0
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
        heston_mu: float, default: 0.0
            Long run average volatility for the Heston model
        heston_vol0: float, default: 0.0
            Starting volatility value for the Heston vol model
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


def convert_to_prices(param: ModelParameters, log_returns: np.ndarray) -> np.ndarray:
    """Converts a sequence of log returns into normal returns (exponentiation)
    and then computes a price sequence given a starting price, param.all_s0.

    Parameters
    ----------
    param: ModelParameters
        Model input
    log_returns:
        Log returns to exponentiate

    Returns
    -------
    numpy.ndarray
        Price series
    """

    returns = np.exp(log_returns)
    # A sequence of prices starting with param.all_s0
    price_sequence: list = [param.all_s0]
    for n in range(1, len(returns)):
        # Add the price at t-1 * return at t
        price_sequence.append(price_sequence[n - 1] * returns[n - 1])
    return np.array(price_sequence)


def brownian_motion_log_returns(
    param: ModelParameters, seed: int | None = None
) -> np.ndarray:
    """This method returns a Wiener process. The Wiener process is also called
    Brownian motion. For more information about the Wiener process check out
    the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process

    Parameters
    ----------
    param: ModelParameters
        Model input
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    numpy.ndarray
        Brownian Motion log returns
    """

    if seed is not None:
        nrand.seed(seed)

    sqrt_delta_sigma = math.sqrt(param.all_delta) * param.all_sigma
    return nrand.normal(loc=0, scale=sqrt_delta_sigma, size=param.all_time)


def brownian_motion_levels(
    param: ModelParameters, seed: int | None = None
) -> np.ndarray:
    """Delivers a price sequence whose returns evolve according to a brownian motion

    Parameters
    ----------
    param: ModelParameters
        Model input
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    numpy.ndarray
        Price sequence which follows a brownian motion
    """

    return convert_to_prices(param, brownian_motion_log_returns(param, seed=seed))


def geometric_brownian_motion_log_returns(
    param: ModelParameters, seed: int | None = None
) -> np.ndarray:
    """This method constructs a sequence of log returns which, when
    exponentiated, produce a random Geometric Brownian Motion (GBM).
    GBM is the stochastic process underlying the Black Scholes
    options pricing formula

    Parameters
    ----------
    param: ModelParameters
        Model input
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    numpy.ndarray
        Log returns of a Geometric Brownian Motion process
    """

    wiener_process = np.array(brownian_motion_log_returns(param, seed=seed))
    sigma_pow_mu_delta = (
        param.gbm_mu - 0.5 * math.pow(param.all_sigma, 2.0)
    ) * param.all_delta
    return wiener_process + sigma_pow_mu_delta


def geometric_brownian_motion_levels(
    param: ModelParameters, seed: int | None = None
) -> np.ndarray:
    """Prices for an asset which evolves according to a geometric brownian motion

    Parameters
    ----------
    param: ModelParameters
        Model input
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    numpy.ndarray
        Price levels for the asset
    """

    return convert_to_prices(
        param, geometric_brownian_motion_log_returns(param, seed=seed)
    )


def jump_diffusion_process(param: ModelParameters, seed: int | None = None) -> list:
    """This method produces a sequence of Jump Sizes which represent a jump
    diffusion process. These jumps are combined with a geometric brownian
    motion (log returns) to produce the Merton model

    Parameters
    ----------
    param: ModelParameters
        Model input
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    numpy.ndarray
        Jump sizes for each point in time (mostly zeroes if jumps are infrequent)
    """

    if seed is not None:
        nrand.seed(seed)
    s_n = time = 0
    small_lamda = -(1.0 / param.lamda)
    jump_sizes = []
    for k in range(0, param.all_time):
        jump_sizes.append(0.0)
    while s_n < param.all_time:
        s_n += small_lamda * math.log(nrand.uniform(0, 1))
        for j in range(0, param.all_time):
            if (
                time * param.all_delta
                <= s_n * param.all_delta
                <= (j + 1) * param.all_delta
            ):
                jump_sizes[j] += nrand.normal(param.jumps_mu, param.jumps_sigma)
                break
        time += 1
    return jump_sizes


def geometric_brownian_motion_jump_diffusion_log_returns(
    param: ModelParameters, seed: int | None = None
) -> np.ndarray:
    """This method constructs combines a geometric brownian motion process
    (log returns) with a jump diffusion process (log returns) to produce a
    sequence of gbm jump returns

    Parameters
    ----------
    param: ModelParameters
        Model input
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    numpy.ndarray
        Geometric Brownian Motion process with jumps in it
    """

    jump_diffusion = jump_diffusion_process(param, seed=seed)
    geometric_brownian_motion = geometric_brownian_motion_log_returns(param, seed=seed)
    return np.add(jump_diffusion, geometric_brownian_motion)


def geometric_brownian_motion_jump_diffusion_levels(
    param: ModelParameters, seed: int | None = None
) -> np.ndarray:
    """Converts returns generated with a Geometric Brownian Motion process
    with jumps into prices

    Parameters
    ----------
    param: ModelParameters
        Model input
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    numpy.ndarray
        Geometric Brownian Motion generated prices
    """

    return convert_to_prices(
        param,
        geometric_brownian_motion_jump_diffusion_log_returns(param, seed=seed),
    )


def heston_construct_correlated_path(
    param: ModelParameters, brownian_motion_one: np.ndarray, seed: int | None = None
) -> (np.ndarray, np.ndarray):
    """This method is a simplified version of the Cholesky decomposition method for just two assets.
    It does not make use of matrix algebra and is therefore quite easy to implement

    Parameters
    ----------
    param: ModelParameters
        Model input
    brownian_motion_one: numpy.ndarray
        A first path to correlate against
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        A correlated Brownian Motion path
    """

    if seed is not None:
        nrand.seed(seed)
    # We do not multiply by sigma here, we do that in the Heston model
    sqrt_delta = math.sqrt(param.all_delta)
    # Construct a path correlated to the first path
    brownian_motion_two = []
    for n in range(param.all_time - 1):
        term_one = param.cir_rho * brownian_motion_one[n]
        term_two = math.sqrt(1 - math.pow(param.cir_rho, 2.0)) * nrand.normal(
            0, sqrt_delta
        )
        brownian_motion_two.append(term_one + term_two)
    return np.array(brownian_motion_one), np.array(brownian_motion_two)


def cox_ingersoll_ross_heston(
    param: ModelParameters, seed: int | None = None
) -> (np.ndarray, np.ndarray):
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
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The interest rate levels for the CIR process
    """

    if seed is not None:
        nrand.seed(seed)

    # We don't multiply by sigma here because we do that in heston
    sqrt_delta_sigma = math.sqrt(param.all_delta) * param.all_sigma
    brownian_motion_volatility = nrand.normal(
        loc=0, scale=sqrt_delta_sigma, size=param.all_time
    )
    a, mu, zero = param.heston_a, param.heston_mu, param.heston_vol0
    volatilities: list = [zero]
    for h in range(1, param.all_time):
        drift = a * (mu - volatilities[-1]) * param.all_delta
        randomness = (
            math.sqrt(max(volatilities[h - 1], 0.05))
            * brownian_motion_volatility[h - 1]
        )
        volatilities.append(max(volatilities[-1], 0.05) + drift + randomness)
    return np.array(brownian_motion_volatility), np.array(volatilities)


def heston_model_levels(
    param: ModelParameters, seed: int | None = None
) -> (np.ndarray, np.ndarray):
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
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The prices for an asset following a Heston process
    """

    brownian, cir_process = cox_ingersoll_ross_heston(param, seed=seed)
    brownian, brownian_motion_market = heston_construct_correlated_path(
        param, brownian, seed=seed
    )

    heston_market_price_levels: list = [param.all_s0]
    for h in range(1, param.all_time):
        drift = param.gbm_mu * heston_market_price_levels[h - 1] * param.all_delta
        vol = (
            cir_process[h - 1]
            * heston_market_price_levels[h - 1]
            * brownian_motion_market[h - 1]
        )
        heston_market_price_levels.append(
            heston_market_price_levels[h - 1] + drift + vol
        )
    return np.array(heston_market_price_levels), np.array(cir_process)


def cox_ingersoll_ross_levels(
    param: ModelParameters, seed: int | None = None
) -> np.ndarray:
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
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The interest rate levels for the CIR process
    """

    brownian_motion = brownian_motion_log_returns(param, seed=seed)
    # Set up the parameters for interest rates
    a, mu, zero = param.cir_a, param.cir_mu, param.all_r0
    # Assumes output is in levels
    levels: list = [zero]
    for h in range(1, param.all_time):
        drift = a * (mu - levels[h - 1]) * param.all_delta
        randomness = math.sqrt(levels[h - 1]) * brownian_motion[h - 1]
        levels.append(levels[h - 1] + drift + randomness)
    return np.array(levels)


def ornstein_uhlenbeck_levels(param: ModelParameters, seed: int | None = None) -> list:
    """This method returns the rate levels of a mean-reverting
    Ornstein Uhlenbeck process

    Parameters
    ----------
    param: ModelParameters
        Model input
    seed: int | None, optional
        Random seed going into numpy.random.seed()

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The interest rate levels for the Ornstein Uhlenbeck process
    """

    ou_levels: list = [param.all_r0]
    brownian_motion_returns = brownian_motion_log_returns(param, seed=seed)
    for h in range(1, param.all_time):
        drift = param.ou_a * (param.ou_mu - ou_levels[h - 1]) * param.all_delta
        randomness = brownian_motion_returns[h - 1]
        ou_levels.append(ou_levels[h - 1] + drift + randomness)
    return ou_levels
