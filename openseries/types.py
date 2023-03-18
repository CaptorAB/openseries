from pydantic import BaseModel
from typing import Literal, TypeVar

Lit_line_plot_mode = Literal[
    "lines",
    "markers",
    "lines+markers",
    "lines+text",
    "markers+text",
    "lines+markers+text",
]
Lit_how_merge = Literal["outer", "inner"]
Lit_quantile_interpolation = Literal["linear", "lower", "higher", "midpoint", "nearest"]
Lit_bizday_frequencies = Literal["BM", "BQ", "BA"]
Lit_pandas_resample_convention = Literal["start", "s", "end", "e"]
Lit_pandas_reindex_method = Literal[
    None, "pad", "ffill", "backfill", "bfill", "nearest"
]
Lit_nan_method = Literal["fill", "drop"]
Lit_capture_ratio = Literal["up", "down", "both"]
Lit_bar_plot_mode = Literal["stack", "group", "overlay", "relative"]
Lit_plotly_output = Literal["file", "div"]

TOpenTimeSeries = TypeVar("TOpenTimeSeries", bound="OpenTimeSeries")
TTestOpenTimeSeries = TypeVar("TTestOpenTimeSeries", bound="TestOpenTimeSeries")
TOpenFrame = TypeVar("TOpenFrame", bound="OpenFrame")
TTestOpenFrame = TypeVar("TTestOpenFrame", bound="TestOpenFrame")
TTestSimPrice = TypeVar("TTestSimPrice", bound="TestSimPrice")
TReturnSimulation = TypeVar("TReturnSimulation", bound="ReturnSimulation")
TTestStochProcesses = TypeVar("TTestStochProcesses", bound="TestStochProcesses")
TTestDateFixer = TypeVar("TTestDateFixer", bound="TestDateFixer")


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

    all_s0: float
    all_time: int
    all_delta: float
    all_sigma: float
    gbm_mu: float
    jumps_lamda: float = 0.0
    jumps_sigma: float = 0.0
    jumps_mu: float = 0.0
    cir_a: float = 0.0
    cir_mu: float = 0.0
    all_r0: float = 0.0
    cir_rho: float = 0.0
    ou_a: float = 0.0
    ou_mu: float = 0.0
    heston_a: float = 0.0
    heston_mu: float = 0.0
    heston_vol0: float = 0.0
