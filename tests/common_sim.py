"""Defining simulated data used in test suite."""

from openseries.simulation import ReturnSimulation, random_generator

SEED: int = 71

ONE_SIM = ReturnSimulation.from_merton_jump_gbm(
    number_of_sims=1,
    trading_days=2512,
    mean_annual_return=0.05,
    mean_annual_vol=0.1,
    jumps_lamda=0.00125,
    jumps_sigma=0.001,
    jumps_mu=-0.2,
    trading_days_in_year=252,
    randomizer=random_generator(seed=SEED),
)

FIVE_SIMS = ReturnSimulation.from_merton_jump_gbm(
    number_of_sims=5,
    trading_days=2512,
    mean_annual_return=0.05,
    mean_annual_vol=0.1,
    jumps_lamda=0.00125,
    jumps_sigma=0.001,
    jumps_mu=-0.2,
    trading_days_in_year=252,
    randomizer=random_generator(seed=SEED),
)
