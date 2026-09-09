"""Phenomenological modified-growth models (background always LCDM).

Implements the two models of Terasawa et al. 2025 (arXiv:2505.09176):

  gamma : f(a) = Omega_m(a)^gamma                       (Linder 2005)
  detg  : alpha(z) = P_L/P_L^LCDM = 1 - beta * (Omega_DE(z)/Omega_DE(0))^p
                                                        (Lin et al. 2024)

Both are expressed as a *linear* power-spectrum rescaling
    P_L(k, a) -> alpha(a) * P_L^LCDM(k, a),   alpha(a) = R(a)^2,
normalized so alpha -> 1 in matter domination (CMB normalization).
The nonlinear mapping (halofit) must be applied AFTER this rescaling
(see hsc_lens.py), matching Terasawa et al.  Never multiply the
nonlinear spectrum by alpha directly.
"""

import numpy as np

GAMMA_LCDM = 0.55


def _E2(a, om):
    return om / a**3 + (1.0 - om)


def _omega_m_a(a, om):
    return (om / a**3) / _E2(a, om)


class GrowthModel:
    """Growth-ratio model on a fixed flat-LCDM background.

    Parameters
    ----------
    model : 'none' | 'gamma' | 'detg'
    om    : Omega_m (total, matching the CCL cosmology)
    par   : gamma (for 'gamma') or beta (for 'detg'); ignored for 'none'
    p     : DETG exponent p (fixed per run; Terasawa et al. use p in 1..6)
    """

    def __init__(self, model, om, par=0.0, p=3.0, a_min=1e-3, n_a=4096):
        self.model = model
        self.om = float(om)
        self.par = float(par)
        self.p = float(p)
        self._a = np.linspace(a_min, 1.0, n_a)
        if model == "gamma":
            self._tabulate_gamma()
        elif model == "detg":
            self._check_detg()
        elif model != "none":
            raise ValueError(f"unknown growth model '{model}'")

    # ---------------- gamma model ----------------
    def _lnD(self, gamma):
        """ln D(a) up to a constant: ln D = int dlna Omega_m(a)^gamma."""
        a = self._a
        integ = _omega_m_a(a, self.om) ** gamma / a
        lnD = np.concatenate([[0.0], np.cumsum(
            0.5 * (integ[1:] + integ[:-1]) * np.diff(a))])
        return lnD

    def _tabulate_gamma(self):
        # R(a) = D_gamma / D_{gamma=0.55}; both normalized deep in matter
        # domination (a_min), where D ~ a for any gamma, so R(a_min)=1.
        self._lnR_tab = self._lnD(self.par) - self._lnD(GAMMA_LCDM)

    # ---------------- DETG model ----------------
    def _x(self, a):
        """Omega_DE(a)/Omega_DE(0) = 1/E^2(a)  (flat LCDM background)."""
        return 1.0 / _E2(a, self.om)

    def _check_detg(self):
        if np.any(self.alpha(self._a) <= 0.0):
            raise ValueError("DETG alpha(a) <= 0; reject this (beta, p).")

    # ---------------- public API ----------------
    def alpha(self, a):
        """P_L / P_L^LCDM = R(a)^2. Scalar in -> scalar out."""
        scalar = np.isscalar(a) or np.ndim(a) == 0
        a = np.atleast_1d(np.asarray(a, dtype=float))
        if self.model == "none":
            out = np.ones_like(a)
        elif self.model == "detg":
            out = 1.0 - self.par * self._x(a) ** self.p
        else:  # gamma
            out = np.exp(2.0 * np.interp(a, self._a, self._lnR_tab))
        return out[0] if scalar else out

    def ratio(self, z):
        """R(z) = D/D_LCDM = sqrt(alpha). Scalar in -> scalar out."""
        return np.sqrt(self.alpha(1.0 / (1.0 + np.asarray(z, dtype=float))))

    def f(self, z, f_lcdm):
        """Modified linear growth rate f(z), given the LCDM f(z).

        gamma:  f = Omega_m(a)^gamma
        detg :  f = f_LCDM + 0.5 dln(alpha)/dln(a)
                  = f_LCDM - 1.5 * beta * p * Omega_m(a) * x^p / alpha
        """
        scalar = np.isscalar(z) or np.ndim(z) == 0
        z = np.atleast_1d(np.asarray(z, dtype=float))
        a = 1.0 / (1.0 + z)
        if self.model == "none":
            out = np.asarray(f_lcdm) * np.ones_like(a)
        elif self.model == "gamma":
            out = _omega_m_a(a, self.om) ** self.par
        else:
            x = self._x(a)
            out = (np.asarray(f_lcdm)
                   - 1.5 * self.par * self.p * _omega_m_a(a, self.om)
                   * x ** self.p / np.atleast_1d(self.alpha(a)))
        return out[0] if scalar else out
