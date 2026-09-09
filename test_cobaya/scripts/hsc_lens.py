"""HSC-Y3 3x2pt (minimal-bias, Sugiyama+2023 format) likelihood for cobaya.

Revision of stone0624/cosmic_S8_project scripts/test_hsc_lens.py.
Conventions below were read off the official pipeline
github.com/HSC-S19A-cosmology-analysis/hscs19a3x2pt-likelihood :

  * PSF additive term (datasetutils.get_psfbias_term):
      psf_pp_pq_qq*.dat has 6 columns [pp+, pp-, pq+, pq-, qq+, qq-]
      on the 30 xi theta bins;
      xi_pm += a^2*pp_pm + 2ab*pq_pm + b^2*qq_pm, added AFTER the
      (1+dm)^2 multiplicative factor.  a=alphapsf, b=betapsf are sampled.
  * dSigma measurement correction (meascorr.dSigma_meascorr_class):
      f_ds = N(dpz, Om_model) / N(0, Om_meas=0.279), where
      N = sum_l sum_j Sigma_cr^{-1}(zl_l, zs_j - dpz) * W[l, j]
      with W read from sumwlssigcritinvPz_z[n].dat
      (cols: idx, zl, then one weight per photoz bin of photoz_bin.dat).
      NOTE the sign: the source grid is shifted zs -> zs - dpz.
  * wp Kaiser correction (gglensing._compute_wp):
      wp = wp_nonlin(pimax) * [ wp_aniso^lin(beta,pimax) / wp_iso^lin(pimax) ]
      i.e. the RSD boost is a ratio built from the LINEAR spectrum
      multipoles (xi0, xi2, xi4; Eq. 48/51 of arXiv:1206.6890), applied
      multiplicatively to the nonlinear real-space wp.  beta = f(z)/b1.
  * Covariance: Hartlap with N_sim = 1404 mock realizations.

Modified growth (gamma / DETG) is applied by rescaling the LINEAR
power spectrum by alpha(a) and letting CCL's halofit act on the
rescaled linear spectrum (ccl.CosmologyCalculator), matching
Terasawa et al. 2025.  The Kaiser beta and the RSD wp use the
modified f(z).

VERIFY-before-MCMC checklist (see CHANGES.md):
  [ ] priors for alphapsf/betapsf, alphamag_i, dpz_0, dm_0, A_IA against
      the released analysis_config.yaml (minimalbias/fiducial).
  [ ] scale cuts reproduce the published masked data-vector length.
  [ ] chi^2 at the published MAP matches chain lnlike to <~0.5.
"""

import os
import numpy as np

# NumPy 2.0 renamed np.trapz -> _trapz; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.special import eval_legendre
from astropy.io import fits
from cobaya.likelihood import Likelihood
import pyccl as ccl

import sys
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "..", "theory"), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from growth_model import GrowthModel  # noqa: E402
except ModuleNotFoundError as _e:
    raise ImportError(
        f"growth_model.py not found. Looked in '{_here}/../theory' and "
        f"'{_here}'. Place it in a 'theory/' folder next to 'scripts/', or "
        f"in the same folder as hsc_lens.py. (Original error: {_e})") from _e

C_OVER_H0 = 2997.92458          # c/H0 in Mpc/h for h=1
OM_MEAS = 0.279                 # cosmology assumed in the measurement
SIGC_CONST = 1.6624e18          # c^2/(4 pi G) in M_sun / Mpc (h=1 units used below)
MNU_FID = 0.06                  # eV, HSC-Y3 fiducial


def E_flat(z, om):
    return np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om))


def chi_hinv_flat(z, om, n=512):
    """Comoving distance in Mpc/h for flat LCDM (vectorized in z)."""
    z = np.atleast_1d(z)
    out = np.empty_like(z, dtype=float)
    for i, zi in enumerate(z):
        zz = np.linspace(0.0, zi, n)
        out[i] = C_OVER_H0 * _trapz(1.0 / E_flat(zz, om), zz)
    return out if out.size > 1 else float(out[0])


def binave_log(func, xlo, xhi, nsub=8):
    """Bin-average func(x) over [xlo, xhi] with weight x dx (annulus),
    i.e. int y x^2 dlnx / int x^2 dlnx, matching the official binave."""
    lx = np.linspace(np.log(xlo), np.log(xhi), nsub)
    x = np.exp(lx)
    w = x ** 2
    y = func(x)
    return _trapz(y * w, lx) / _trapz(w, lx)


class SigmaCritCorrection:
    """Pair-weighted <Sigma_cr^{-1}> measurement correction, per lens bin.

    Port of meascorr.dSigma_meascorr_class (direct mode).  Distances are
    computed in h-inverse units; h cancels in the ratio.
    """

    def __init__(self, fname_weights, fname_zsbin):
        data = np.loadtxt(fname_weights)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.zl = data[:, 1]
        self.W = data[:, 2:]                       # (N_lens, N_zsbin)
        self.zs = np.loadtxt(fname_zsbin)
        assert self.W.shape[1] == self.zs.size, \
            "sumwlssigcritinvPz / photoz_bin size mismatch"
        self._denom = self._numerator(0.0, OM_MEAS)

    def _numerator(self, dpz, om):
        zs = self.zs - dpz                          # official sign convention
        chi_l = chi_hinv_flat(self.zl, om)
        chi_s = chi_hinv_flat(np.clip(zs, 0.0, None), om)
        chi_s[zs <= 0.0] = 0.0
        num = 0.0
        for j in range(zs.size):
            if chi_s[j] <= 0.0:
                continue
            sci = (1.0 + self.zl) * chi_l * (1.0 - chi_l / chi_s[j])
            sci[sci < 0.0] = 0.0
            num += np.sum(sci * self.W[:, j])
        return num

    def f_ds(self, dpz, om):
        return self._numerator(dpz, om) / self._denom


class HSC_Lens(Likelihood):
    """3x2pt likelihood: dSigma (3 lens bins) + xi_pm + wp (3 lens bins)."""

    # ---- configurable via yaml ----
    data_folder: str = ""
    dataset_file: str = "dataset_hsc_y3.fits"
    psf_file: str = ""                    # dataset/psf_pp_pq_qq_used.dat
    sumwlssig_files: list = []            # [dataset/sumwlssigcritinvPz_z0.dat, ...]
    zsbin_file: str = ""                  # dataset/photoz_bin.dat
    growth_model: str = "none"            # 'none' | 'gamma' | 'detg'
    p_detg: float = 3.0
    n_sim: int = 1404                     # Hartlap (More+2023 mocks)
    binave: bool = True
    pi_max: float = 100.0
    z_l_eff: list = [0.2607, 0.5106, 0.6264]
    # scale cuts (h^-1 Mpc for ds/wp, arcmin for xi) -- VERIFY vs published
    ds_rmin: float = 12.0
    ds_rmax: list = [30.0, 40.0, 80.0]
    wp_rmin: float = 8.0
    wp_rmax: float = 80.0
    xip_tmin: float = 7.1
    xip_tmax: float = 56.6
    xim_tmin: float = 31.2
    xim_tmax: float = 248.0

    output_params = ["sigma8", "S8", "S8_z_L1", "S8_z_L2", "S8_z_L3"]

    # ------------------------------------------------------------------ init
    def initialize(self):
        path = os.path.join(self.data_folder, self.dataset_file)
        self.log.info(f"Loading dataset: {path}")
        with fits.open(path) as hdul:
            self.ds_t = hdul["ds"].data
            self.xip_t = hdul["xip"].data
            self.xim_t = hdul["xim"].data
            self.wp_t = hdul["wp"].data
            full_cov = np.array(hdul["COVMAT"].data, dtype=float)
            self.z_s = hdul["nz_source"].data["Z_MID"]
            self.nz_s = hdul["nz_source"].data["BIN1"]
            self.z_lg = hdul["nz_lens"].data["Z_MID"]
            self.nz_l = [hdul["nz_lens"].data[f"BIN{i+1}"] for i in range(3)]
        # sanity: lens n(z) must cover LOWZ/CMASS ranges
        for i, (lo, hi) in enumerate([(0.10, 0.36), (0.42, 0.56), (0.54, 0.71)]):
            zmax = self.z_lg[self.nz_l[i] > 0].max()
            if not (lo < zmax < hi + 0.3):
                self.log.warning(
                    f"nz_lens BIN{i+1} peaks at z<={zmax:.2f}; expected "
                    f"~[{lo},{hi}]. Rebuild the FITS from nzl_z{i}_100bin.dat!")

        # masks
        self.ds_cut = np.zeros(len(self.ds_t), dtype=bool)
        for b, rmax in enumerate(self.ds_rmax, start=1):
            self.ds_cut |= ((self.ds_t["BIN1"] == b)
                            & (self.ds_t["ANG"] >= self.ds_rmin)
                            & (self.ds_t["ANG"] <= rmax))
        self.wp_cut = ((self.wp_t["ANG"] >= self.wp_rmin)
                       & (self.wp_t["ANG"] <= self.wp_rmax))
        self.xip_cut = ((self.xip_t["ANG"] >= self.xip_tmin)
                        & (self.xip_t["ANG"] <= self.xip_tmax))
        self.xim_cut = ((self.xim_t["ANG"] >= self.xim_tmin)
                        & (self.xim_t["ANG"] <= self.xim_tmax))

        self.data_vector = np.concatenate([
            self.ds_t["VALUE"][self.ds_cut],
            self.xip_t["VALUE"][self.xip_cut],
            self.xim_t["VALUE"][self.xim_cut],
            self.wp_t["VALUE"][self.wp_cut]])
        n = len(self.data_vector)

        off = np.cumsum([0, len(self.ds_t), len(self.xip_t), len(self.xim_t)])
        keep = np.concatenate([
            np.where(self.ds_cut)[0] + off[0],
            np.where(self.xip_cut)[0] + off[1],
            np.where(self.xim_cut)[0] + off[2],
            np.where(self.wp_cut)[0] + off[3]])
        cov = full_cov[np.ix_(keep, keep)]
        hartlap = (self.n_sim - n - 2) / (self.n_sim - 1)
        self.inv_cov = np.linalg.inv(cov) * hartlap
        self.log.info(f"data vector: ds={self.ds_cut.sum()} "
                      f"xip={self.xip_cut.sum()} xim={self.xim_cut.sum()} "
                      f"wp={self.wp_cut.sum()} total={n} "
                      f"(Hartlap {hartlap:.4f}, Nsim={self.n_sim})")

        # PSF templates (REQUIRED for the fiducial analysis)
        if self.psf_file:
            self.psf = np.loadtxt(os.path.join(self.data_folder, self.psf_file))
            assert self.psf.shape[1] == 6, \
                "psf file must have 6 cols: pp+ pp- pq+ pq- qq+ qq-"
            assert self.psf.shape[0] == len(self.xip_t), \
                "psf rows must match xi theta bins"
        else:
            self.log.warning("No psf_file: PSF additive term set to 0. "
                             "The fiducial Sugiyama+ analysis REQUIRES it.")
            self.psf = np.zeros((len(self.xip_t), 6))

        # pair-weighted Sigma_crit correction (REQUIRED for fiducial)
        self.meascorr = None
        if self.sumwlssig_files and self.zsbin_file:
            zsbin = os.path.join(self.data_folder, self.zsbin_file)
            self.meascorr = [SigmaCritCorrection(
                os.path.join(self.data_folder, f), zsbin)
                for f in self.sumwlssig_files]
            self.log.info("Using pair-weighted sumwlssigcritinvPz correction.")
        else:
            self.log.warning("No sumwlssig files: falling back to stacked-p(z)"
                             " <Sigma_cr^-1>. Expect %-level dSigma offsets "
                             "growing with lens z (cf. CMASS2).")

    # ------------------------------------------------------------ cobaya API
    def get_requirements(self):
        req = {p: None for p in
               ["ombh2", "omch2", "H0", "ns", "As",
                "b1", "b2", "b3", "AIA", "dm_0", "dpz_0",
                "alphapsf", "betapsf",
                "alphamag_1", "alphamag_2", "alphamag_3"]}
        if self.growth_model == "gamma":
            req["gamma_growth"] = None
        elif self.growth_model == "detg":
            req["beta_detg"] = None
        return req

    # ------------------------------------------------------------ cosmology
    def _build_cosmology(self, p):
        h = p["H0"] / 100.0
        base = ccl.Cosmology(
            Omega_c=p["omch2"] / h**2, Omega_b=p["ombh2"] / h**2, h=h,
            A_s=p["As"], n_s=p["ns"], m_nu=MNU_FID,
            transfer_function="boltzmann_camb",
            matter_power_spectrum="halofit")
        om = base["Omega_m"]

        gpar = (p.get("gamma_growth") if self.growth_model == "gamma"
                else p.get("beta_detg") if self.growth_model == "detg"
                else 0.0)
        gr = GrowthModel(self.growth_model, om, par=gpar or 0.0, p=self.p_detg)

        if self.growth_model == "none":
            return base, base, gr, om
        # rescale LINEAR pk, then halofit on the rescaled linear spectrum
        a_arr = np.linspace(1.0 / (1.0 + 6.0), 1.0, 48)
        k_arr = np.logspace(-4, 1.7, 384)          # 1/Mpc
        pk = np.array([ccl.linear_matter_power(base, k_arr, a) for a in a_arr])
        pk *= gr.alpha(a_arr)[:, None]
        calc = ccl.CosmologyCalculator(
            Omega_c=p["omch2"] / h**2, Omega_b=p["ombh2"] / h**2, h=h,
            A_s=p["As"], n_s=p["ns"], m_nu=MNU_FID,
            pk_linear={"a": a_arr, "k": k_arr,
                       "delta_matter:delta_matter": pk},
            nonlinear_model="halofit")
        return base, calc, gr, om

    # -------------------------------------------------------------- theory
    def theory_vector(self, base, cosmo, gr, om, p):
        h = base["h"]
        b = [p["b1"], p["b2"], p["b3"]]
        dm, dpz = p["dm_0"], p["dpz_0"]
        amag = [p["alphamag_1"], p["alphamag_2"], p["alphamag_3"]]
        rho_m0 = ccl.rho_x(base, 1.0, "matter", is_comoving=True)  # Msun/Mpc^3
        # shifted source n(z): effective zs -> zs - dpz  <=>  n(z) -> n(z+dpz)
        nz_true = np.interp(self.z_s + dpz, self.z_s, self.nz_s,
                            left=0.0, right=0.0)

        pk_nl = cosmo.get_nonlin_power()
        pk_lin = cosmo.get_linear_power()

        m_ds, m_wp = [], []
        for i in range(3):
            zl = self.z_l_eff[i]
            a_l = 1.0 / (1.0 + zl)
            # ---- measurement corrections (cosmology + photo-z) ----
            chi_C, chi_ref = chi_hinv_flat(zl, om), chi_hinv_flat(zl, OM_MEAS)
            R_fac = chi_C / chi_ref                    # r_corr
            E_fac = E_flat(zl, om) / E_flat(zl, OM_MEAS)
            pimax_wp = self.pi_max / E_fac
            if self.meascorr is not None:
                f_ds = self.meascorr[i].f_ds(dpz, om)
            else:  # stacked-p(z) fallback (approximate)
                f_ds = (self._sci_mean(zl, om, nz_true)
                        / self._sci_mean(zl, OM_MEAS, self.nz_s))

            mask_ds = (self.ds_t["BIN1"] == i + 1) & self.ds_cut
            mask_wp = (self.wp_t["BIN1"] == i + 1) & self.wp_cut
            edges_ds = list(zip(self.ds_t["ANGLEMIN"][mask_ds],
                                self.ds_t["ANGLEMAX"][mask_ds]))
            edges_wp = list(zip(self.wp_t["ANGLEMIN"][mask_wp],
                                self.wp_t["ANGLEMAX"][mask_wp]))
            rp_ds_c = R_fac * self.ds_t["ANG"][mask_ds]

            # ---- 3D correlation functions from the (rescaled) spectra ----
            r_grid = np.geomspace(1e-2, 300.0, 600)           # Mpc/h
            xi_nl = ccl.correlation_3d(base, a=a_l, r=r_grid / h,
                                       p_of_k_a=pk_nl)
            xi_gm = b[i] * xi_nl
            xi_gg = b[i] ** 2 * xi_nl

            # ---- dSigma: minimal-bias Abel integrals + PM-free model ----
            pi_arr = np.linspace(0.0, self.pi_max, 300)

            def Sigma_of(rp):
                r3 = np.sqrt(rp[:, None] ** 2 + pi_arr[None, :] ** 2)
                x = np.interp(r3, r_grid, xi_gm)
                return 2 * rho_m0 * _trapz(x, pi_arr, axis=1) / (h**2 * 1e12)

            def ds_at(rp):
                rp = np.atleast_1d(rp)
                Sig = Sigma_of(rp)
                out = np.empty_like(rp)
                for j, r0 in enumerate(rp):
                    rin = np.linspace(1e-3, r0, 120)
                    Sin = Sigma_of(rin)
                    out[j] = 2 / r0**2 * _trapz(Sin * rin, rin) - Sig[j]
                return out

            # magnification (uses shifted nz and the rescaled power)
            kern = self._sci_mean(zl, om, nz_true) / h        # Mpc, h=1 units
            Sig_c = SIGC_CONST / ((1.0 + zl) * kern)
            tr_l = ccl.WeakLensingTracer(base, dndz=(self.z_lg, self.nz_l[i]))
            tr_s = ccl.WeakLensingTracer(base, dndz=(self.z_s, nz_true))
            ell = np.geomspace(0.1, 1e5, 512)
            cl_ls = ccl.angular_cl(base, tr_l, tr_s, ell=ell, p_of_k_a=pk_nl)

            def ds_mag_at(rp):
                th = (R_fac * 0 + rp) / chi_C * (180.0 / np.pi)  # deg
                gt = ccl.correlation(base, ell=ell, C_ell=cl_ls, theta=th,
                                     type="NG", method="FFTLog")
                return 2.0 * (amag[i] - 1.0) * Sig_c * gt / (h * 1e12)

            def ds_model(rp):
                return ds_at(rp) + ds_mag_at(rp)

            if self.binave:
                vals = [binave_log(ds_model, R_fac * lo, R_fac * hi)
                        for lo, hi in edges_ds]
            else:
                vals = ds_model(rp_ds_c)
            m_ds.extend([(1.0 + dm) * f_ds * v for v in np.atleast_1d(vals)])

            # ---- wp: nonlinear wp x LINEAR Kaiser ratio (official) ----
            f_lcdm = ccl.growth_rate(base, a_l)
            f_mod = float(gr.f(zl, f_lcdm))
            beta = f_mod / b[i]
            xi_lin = ccl.correlation_3d(base, a=a_l, r=r_grid / h,
                                        p_of_k_a=pk_lin) * b[i] ** 2
            kaiser = _kaiser_ratio(r_grid, xi_lin, beta, pimax_wp)

            def wp_nl(rp):
                r3 = np.sqrt(rp[:, None] ** 2
                             + np.linspace(0, pimax_wp, 300)[None, :] ** 2)
                x = np.interp(r3, r_grid, xi_gg)
                return 2 * _trapz(x, np.linspace(0, pimax_wp, 300),
                                        axis=1)

            def wp_model(rp):
                return wp_nl(np.atleast_1d(rp)) * kaiser(rp)

            if self.binave:
                vals = [binave_log(wp_model, R_fac * lo, R_fac * hi)
                        for lo, hi in edges_wp]
            else:
                vals = wp_model(R_fac * self.wp_t["ANG"][mask_wp])
            m_wp.extend([E_fac * v for v in np.atleast_1d(vals)])

        # ---- cosmic shear xi_pm: NLA IA, x(1+dm)^2, + PSF term ----
        ia = (self.z_s, np.full_like(self.z_s, float(p["AIA"])))
        tr = ccl.WeakLensingTracer(base, dndz=(self.z_s, nz_true), ia_bias=ia)
        ell = np.geomspace(0.1, 1e5, 1024)
        cl = ccl.angular_cl(base, tr, tr, ell=ell, p_of_k_a=pk_nl)
        fac_m = (1.0 + dm) ** 2
        a_psf, b_psf = p["alphapsf"], p["betapsf"]

        def xi_pm(which, table, cut, cols):
            th = table["ANG"][cut] / 60.0
            val = fac_m * ccl.correlation(base, ell=ell, C_ell=cl, theta=th,
                                          type=which, method="FFTLog")
            pp, pq, qq = (self.psf[cut, c] for c in cols)
            return val + a_psf**2 * pp + 2*a_psf*b_psf * pq + b_psf**2 * qq

        xip = xi_pm("GG+", self.xip_t, self.xip_cut, (0, 2, 4))
        xim = xi_pm("GG-", self.xim_t, self.xim_cut, (1, 3, 5))

        return np.concatenate([m_ds, xip, xim, m_wp])

    def _sci_mean(self, zl, om, nz):
        chi_l = chi_hinv_flat(zl, om)
        chi_s = chi_hinv_flat(self.z_s, om)
        integ = np.where(self.z_s > zl,
                         chi_l * (chi_s - chi_l) / np.where(chi_s > 0, chi_s, 1.0),
                         0.0) * nz
        return _trapz(integ, self.z_s) / _trapz(nz, self.z_s)

    # ---------------------------------------------------------------- logp
    def logp(self, _derived=None, **pv):
        base, cosmo, gr, om = self._build_cosmology(pv)
        m = self.theory_vector(base, cosmo, gr, om, pv)
        if len(m) != len(self.data_vector):
            raise ValueError(f"theory={len(m)} vs data={len(self.data_vector)}")
        d = self.data_vector - m
        chi2 = d @ self.inv_cov @ d
        if _derived is not None:
            s8 = ccl.sigma8(base)          # LCDM (CMB-normalized) sigma8
            _derived["sigma8"] = s8
            _derived["S8"] = s8 * np.sqrt(om / 0.3)
            for i, zl in enumerate(self.z_l_eff):
                D = ccl.growth_factor(base, 1.0 / (1.0 + zl))
                _derived[f"S8_z_L{i+1}"] = (s8 * float(gr.ratio(zl)) * D
                                            * np.sqrt(om / 0.3))
        return -0.5 * chi2


def _kaiser_ratio(r, xi_lin, beta, pimax):
    """Return a function rp -> wp_aniso^lin/wp_iso^lin (Eq. 48/51 of
    arXiv:1206.6890), i.e. the official multiplicative Kaiser boost."""
    from scipy.integrate import cumulative_trapezoid
    J3 = cumulative_trapezoid(xi_lin * r**2, r, initial=0) / r**3
    J5 = cumulative_trapezoid(xi_lin * r**4, r, initial=0) / r**5
    xi0 = (1 + 2/3*beta + 1/5*beta**2) * xi_lin
    xi2 = (4/3*beta + 4/7*beta**2) * (xi_lin - 3*J3)
    xi4 = 8/35*beta**2 * (xi_lin + 15/2*J3 - 35/2*J5)
    s0, s2, s4 = ius(r, xi0), ius(r, xi2), ius(r, xi4)
    s_iso = ius(r, xi_lin)
    rpi = np.logspace(-3, np.log10(pimax), 300)

    def ratio(rp):
        rp = np.atleast_1d(rp)
        s = np.sqrt(rp[:, None]**2 + rpi[None, :]**2)
        mu = rpi[None, :] / s
        num = _trapz(s0(s) + s2(s)*eval_legendre(2, mu)
                           + s4(s)*eval_legendre(4, mu), rpi, axis=1)
        den = _trapz(s_iso(s), rpi, axis=1)
        return num / den
    return ratio
