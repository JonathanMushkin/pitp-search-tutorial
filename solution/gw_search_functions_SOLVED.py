from typing import Tuple, Optional, Union, List
import numpy as np
from scipy import stats
from scipy import interpolate
import numba

# Constants
MTSUN_SI = 4.925491025543576e-06  # solar mass in seconds
MCHIRP_RANGE = (1.15, 1.2)  # in solar masses
MASS_RATIO_RANGE = (0.74, 1)


######################################################################
# Functions given for the exam, that deal with Post Newtonian physics
# of the waveform


def phases_to_linear_free_phases(
    phases: np.ndarray, freqs: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Convert phases to linear-free phases by removing linear frequency evolution.

    Args:
        phases: Array of phases (2D array: n_phases x n_frequencies)
        freqs: Array of frequencies (Hz)
        weights: Array of weights for frequency weighting

    Returns:
        Array of linear-free phases with same shape as input phases
    """
    # find first and second moments of the frequency distribution
    f_bar = np.sum(freqs * weights**2)
    f2_bar = np.sum(freqs**2 * weights**2)
    sigma_f = np.sqrt(f2_bar - f_bar**2)
    # define the linear component
    psi_1 = (freqs - f_bar) / sigma_f
    # find the linear component projection on each phase
    c_1 = np.sum(phases * weights**2 * psi_1, axis=-1)
    # remove linear component from each phase
    phases_shifted = phases - np.outer(c_1, psi_1)
    # fix phase at f_bar to be zero
    phases_shifted -= np.array(
        [np.interp(x=f_bar, xp=freqs, fp=p) for p in phases_shifted]
    )[..., None]
    return phases_shifted


def m1m2_to_mchirp(
    m1: Union[float, np.ndarray], m2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Return chirp mass given component masses.

    Args:
        m1: Primary mass in solar masses
        m2: Secondary mass in solar masses

    Returns:
        Chirp mass in solar masses
    """
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def mchirp_q_to_m1m2(
    mchirp: Union[float, np.ndarray], q: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Return component masses given chirp mass and mass ratio.

    Args:
        mchirp: Chirp mass in solar masses
        q: Mass ratio q = m2/m1 (where m2 <= m1)

    Returns:
        Tuple of (m1, m2) component masses in solar masses
    """
    m1 = mchirp * (1 + q) ** 0.2 / q**0.6
    m2 = q * m1
    return m1, m2


def masses_to_pn_coefficients(
    m1: Union[float, np.ndarray], m2: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Return the post-Newtonian coefficients for the phase of a waveform
    given the masses of the binary.

    Args:
        m1: Primary mass(es) in solar masses
        m2: Secondary mass(es) in solar masses

    Returns:
        Array of post-Newtonian coefficients (last axis is PN order)
    """
    (
        m1,
        m2,
    ) = np.broadcast_arrays(m1, m2)
    mtot = m1 + m2
    eta = m1 * m2 / mtot**2
    q = m2 / m1
    chis = 0.0
    chia = 0.0
    sx = 0.0
    sy = 0.0
    delta = (m1 - m2) / mtot
    beta = 113 / 12 * (chis + delta * chia - 76 / 113 * eta * chis)

    pn_coefficients = (
        3 / 128 / eta * mtot ** (-5 / 3),
        3 / 128 * (55 / 9 + 3715 / 756 / eta) / mtot,
        3 / 128 * (4 * beta - 16 * np.pi) / eta * mtot ** (-2 / 3),
        15
        / 128
        * (1 + q) ** 4
        * (4 / 3 + q)
        * (sx**2 + sy**2)
        / mtot**4
        / q**2
        * mtot ** (-1 / 3),
    )

    return np.moveaxis(pn_coefficients, 0, -1)  # Last axis is PN order


def pn_coefficients_to_phases(
    freqs: np.ndarray, pn_coefficients: np.ndarray
) -> np.ndarray:
    """
    Return the phase of a waveform given the post-Newtonian coefficients.

    Args:
        freqs: Array of frequencies (Hz)
        pn_coefficients: Array of post-Newtonian coefficients

    Returns:
        Array of phases (same shape as freqs or 2D if multiple coefficient sets)
    """
    freqs = np.atleast_1d(freqs)
    i = np.where(freqs > 0)[0][0]
    powers = [-5 / 3, -3 / 3, -2 / 3, -1 / 3]
    pn_functions = np.zeros((len(freqs), len(powers)))
    pn_functions[i:,] = -np.power.outer(MTSUN_SI * np.pi * freqs[i:], powers)

    # pn_functions.shape = (len(freqs), len(powers))
    if pn_coefficients.ndim == 1:
        # return a single phase
        phase = pn_functions @ pn_coefficients
    else:
        # return a phase for each set of coefficients, (n_samples, n_freqs)
        phase = pn_coefficients @ pn_functions.T

    return phase


def masses_to_phases(
    m1: Union[float, np.ndarray],
    m2: Union[float, np.ndarray],
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Return the phase of a waveform given the masses of the binary.

    Args:
        m1: Primary mass(es) in solar masses
        m2: Secondary mass(es) in solar masses
        freqs: Array of frequencies (Hz)

    Returns:
        2D array of phases (per mass pair, per frequency)
    """
    return pn_coefficients_to_phases(freqs, masses_to_pn_coefficients(m1, m2))


def draw_mass_samples(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw n_samples of primary and secondary masses from a fiducial mass distribution.

    Uses chirp-mass and mass ratio as uniform random variables and transforms
    them to the primary and secondary masses.

    Args:
        n_samples: Number of mass samples to draw

    Returns:
        Tuple of (m1_samples, m2_samples) arrays of primary and secondary masses
    """

    u = stats.qmc.Halton(2).random(n_samples)
    mchirp_samples = stats.uniform(MCHIRP_RANGE[0], np.diff(MCHIRP_RANGE)).ppf(
        u[:, 0]
    )
    q_samples = stats.uniform(
        MASS_RATIO_RANGE[0], np.diff(MASS_RATIO_RANGE)
    ).ppf(u[:, 1])
    m1_samples, m2_samples = mchirp_q_to_m1m2(mchirp_samples, q_samples)
    return m1_samples, m2_samples


##
# Functions skeletons to be completed in the exam
##


def correlate(
    x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return the correlation of two time series.

    Using common time convention, x is the data and y is the template.

    Args:
        x: Frequency domain data series
        y: Frequency domain template series
        w: Whitening filter (optional)

    Returns:
        Array of correlation values in time domain
    """
    # if w is not None, multiply x and y by the whitening filter
    if w is None:
        return np.fft.irfft(x * y.conj())
    else:
        return np.fft.irfft((x * w) * (y * w).conj())


def select_points_without_clutter(
    points: np.ndarray, distance_scale: float
) -> Tuple[np.ndarray, List[int]]:
    """
    Select a subset of the points such that no two points are closer than distance_scale.

    The function iterates over the points and adds a new point to the subset if it is
    further than distance_scale from all points in the subset.

    Args:
        points: Array of points (n_points x n_dimensions)
        distance_scale: Minimum distance between any two points in the subset

    Returns:
        Tuple of (subset_points, indices) where indices are the original indices
    """
    subset = []
    indices = []
    for i, point in enumerate(points):
        # Check if point is far enough from all existing points
        too_close = False
        for subset_point in subset:
            if np.linalg.norm(point - subset_point) < distance_scale:
                too_close = True
                break  # No need to check other points once we find one too close

        if not too_close:
            subset.append(point)
            indices.append(i)
    return np.array(subset), indices


def complex_overlap_timeseries(
    template: np.ndarray, data: np.ndarray
) -> np.ndarray:
    """
    Return a complex time series of the SNR of a template in a data stream.

    Args:
        template: Frequency domain template (whitened)
        data: Frequency domain data (whitened)

    Returns:
        Array of complex SNR time series
    """
    z_cos = correlate(data, template)
    z_sin = correlate(data, template * 1j)

    return z_cos + 1j * z_sin


def snr2_timeseries(template: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Return a time series of the SNR squared of a template in a data stream.

    Args:
        template: Frequency domain template (whitened)
        data: Frequency domain data (whitened)

    Returns:
        Array of (real) SNR^2 time series
    """
    return np.abs(complex_overlap_timeseries(template, data)) ** 2


@numba.njit()
def max_argmax_over_n_samples(
    x: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the maximum and argmax of x for each segment of length n.

    Args:
        x: Input array
        n: Length of each segment

    Returns:
        Tuple of (maxs, argmaxs) arrays
    """
    length = len(x)
    maxs = np.zeros(length // n)
    argmaxs = np.zeros(length // n, dtype=np.int64)
    for i in range(length // n):
        start = i * n
        end = min((i + 1) * n, length)
        maxs[i] = np.max(x[start:end])
        argmaxs[i] = np.argmax(x[start:end]) + start

    return maxs, argmaxs
