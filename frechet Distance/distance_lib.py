import numpy as np
import math
from numba import jit


@jit(nopython=True, fastmath=True)
def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(np.dot(d, d))


@jit(nopython=True, fastmath=True)
def haversine(p: np.ndarray,
              q: np.ndarray) -> float:
    """
    Vectorized haversine distance calculation
    :p: Initial location in radians
    :q: Final location in radians
    :return: Distance
    """
    d = q - p
    a = math.sin(d[0]/2.0)**2 + math.cos(p[0]) * math.cos(q[0]) \
        * math.sin(d[1]/2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return c


@jit(nopython=True, fastmath=True)
def earth_haversine(p: np.ndarray, q: np.ndarray) -> float:
    """
    Vectorized haversine distance calculation
    :p: Initial location in degrees [lat, lon]
    :q: Final location in degrees [lat, lon]
    :return: Distances in meters
    """
    earth_radius = 6378137.0
    return haversine(np.radians(p), np.radians(q)) * earth_radius