from __future__ import annotations

def u_shape_density_factory(a: float = 2.0):
    """
    Return phi(x) for x in [0,1], proportional to (x(1-x))^(a-1).
    No need to normalize; the integral cancels in VWAP.
    """
    def phi(x: float) -> float:
        x = max(0.0, min(x, 1.0))
        return (x * (1.0 - x)) ** (a - 1.0)
    return phi
