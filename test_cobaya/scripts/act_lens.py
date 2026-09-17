import numpy as np
import pyccl as ccl
import act_dr6_lenslike as alike

import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "..", "theory"), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from growth_model import GrowthModel

class CCLState:
    """Everything the low-z likelihoods need, built once per MCMC step."""
 
    __slots__ = ("base", "calc", "growth", "omega_m", "h", "pk_nl", "pk_lin")
 
    def __init__(self, base, calc, growth, omega_m):
        self.base = base            # pure-LCDM ccl.Cosmology (distances, sigma8)
        self.calc = calc            # DETG-rescaled cosmology (== base if 'none')
        self.growth = growth        # GrowthModel
        self.omega_m = omega_m
        self.h = base["h"]
        self.pk_nl = calc.get_nonlin_power()
        self.pk_lin = calc.get_linear_power()

    def build_ccl(p, growth_model="none", p_detg=3.0, m_nu=0.06,
              mass_split="normal", z_nl=6.0, n_a_nl=48,
              lk_min=-4.0, lk_max=1.7, nk=384):
        """Build the (base, DETG-rescaled) CCL pair for one parameter point.
    
        `p` is a dict with ombh2, omch2, H0, ns, As and, if needed,
        gamma_growth / beta_detg.  Returns (CCLState, k_arr, a_nl) or None if
        the growth parameters are unphysical.
        """
        h = p["H0"] / 100.0
        kw = dict(Omega_c=p["omch2"] / h**2, Omega_b=p["ombh2"] / h**2, h=h,
                A_s=p["As"], n_s=p["ns"], m_nu=m_nu, mass_split=mass_split)
    
        base = ccl.Cosmology(transfer_function="boltzmann_camb",
                            matter_power_spectrum="halofit", **kw)
        om = base["Omega_m"]
    
        gpar = (p.get("gamma_growth") if growth_model == "gamma"
                else p.get("beta_detg") if growth_model == "detg" else 0.0)
        try:
            gr = GrowthModel(growth_model, om, par=gpar or 0.0, p=p_detg)
        except ValueError:
            return None
        if not gr.valid():
            return None
    
        a_nl = np.linspace(1.0 / (1.0 + z_nl), 1.0, n_a_nl)
        k_arr = np.logspace(lk_min, lk_max, nk)            # 1/Mpc
    
        if growth_model == "none":
            return CCLState(base, base, gr, om), k_arr, a_nl
    
        # rescale the LINEAR pk, then let halofit act on the rescaled spectrum
        pk = np.array([ccl.linear_matter_power(base, k_arr, a) for a in a_nl])
        pk *= gr.alpha(a_nl)[:, None]
        calc = ccl.CosmologyCalculator(
            pk_linear={"a": a_nl, "k": k_arr, "delta_matter:delta_matter": pk},
            nonlinear_model="halofit", **kw)
        return CCLState(base, calc, gr, om), k_arr, a_nl

        
    def kk_transfer_ratio(st, k_arr, a_nl, lmax=4000, z_max=1200.0, n_a_hi=48,
                        n_ell=48, z_star=1089.9, l_limber=-1):
        """r(L) = C_L^kk[DETG] / C_L^kk[LCDM], as an array on ell = 0..lmax.
    
        Returns ones for growth_model == 'none'.
        """
        def _hybrid_pk2d(cosmo_for_lin, k_arr, a_nl, a_hi, alpha_hi, pk_nl_lo):
                """Nonlinear P(k,a) over the full lensing range.
            
                z <= z_nl : halofit (already computed on the rescaled linear spectrum)
                z >  z_nl : linear * alpha(a).  Halofit's correction there is <0.1%
                            for the k that matter, and -- crucially -- the SAME recipe
                            is used for the LCDM reference, so it cancels in r(L).
                """
                hi = np.array([ccl.linear_matter_power(cosmo_for_lin, k_arr, a)
                            for a in a_hi]) * alpha_hi[:, None]
                pk = np.concatenate([hi, pk_nl_lo], axis=0)
                a = np.concatenate([a_hi, a_nl])
                return ccl.Pk2D(a_arr=a, lk_arr=np.log(k_arr), pk_arr=np.log(pk),
                                is_logp=True, extrap_order_lok=1, extrap_order_hik=2)

        ell_out = np.arange(lmax + 1)
        if st.growth.model == "none":
            return np.ones(ell_out.size)
    
        a_hi = np.geomspace(1.0 / (1.0 + z_max), a_nl[0], n_a_hi, endpoint=False)
        gr = st.growth
        al_hi, one_hi = gr.alpha(a_hi), np.ones_like(a_hi)
    
        pk_lo_mod = np.array([ccl.nonlin_matter_power(st.calc, k_arr, a)
                            for a in a_nl])
        pk_lo_ref = np.array([ccl.nonlin_matter_power(st.base, k_arr, a)
                            for a in a_nl])
        pk_mod = _hybrid_pk2d(st.base, k_arr, a_nl, a_hi, al_hi, pk_lo_mod)
        pk_ref = _hybrid_pk2d(st.base, k_arr, a_nl, a_hi, one_hi, pk_lo_ref)
    
        kw_mod = dict(p_of_k_a=pk_mod, l_limber=l_limber)
        kw_ref = dict(p_of_k_a=pk_ref, l_limber=l_limber)
        if l_limber >= 0:
            # non-Limber (FKEM) also needs a LINEAR Pk2D of the same type
            ll_mod = np.array([ccl.linear_matter_power(st.calc, k_arr, a)
                            for a in a_nl])
            ll_ref = np.array([ccl.linear_matter_power(st.base, k_arr, a)
                            for a in a_nl])
            kw_mod["p_of_k_a_lin"] = _hybrid_pk2d(st.base, k_arr, a_nl, a_hi,
                                                al_hi, ll_mod)
            kw_ref["p_of_k_a_lin"] = _hybrid_pk2d(st.base, k_arr, a_nl, a_hi,
                                                one_hi, ll_ref)
    
        tr = ccl.CMBLensingTracer(st.base, z_source=z_star)
        nodes = np.unique(np.geomspace(8.0, max(lmax, 10), n_ell).astype(int))
        c_mod = ccl.angular_cl(st.base, tr, tr, nodes, **kw_mod)
        c_ref = ccl.angular_cl(st.base, tr, tr, nodes, **kw_ref)
        r = c_mod / c_ref
        return np.interp(ell_out, nodes, r, left=r[0], right=r[-1])


class ACTDR6LensDETG(alike.ACTDR6LensLike):

    # ---- new options (everything else is inherited: variant, lens_only,
    #      trim_lmax, lmax, apply_hartlap, varying_cmb_alens, limber, ...) ----
    apply_growth_ratio: bool = True
    ddir: str = ""          # explicit path to the extracted v1.2 data dir
    growth_model: str = "detg"
    p_detg: float = 3.0
    m_nu: float = 0.06
    mass_split: str = "normal"

    def initialize(self):
        if not self.ddir:
            super().initialize()
        else:
            # mirror of upstream initialize(), with the data dir forced
            if self.lens_only:
                self.no_like_corrections = True
            if self.lmax < self.trim_lmax:
                raise ValueError(f"lmax >= {self.trim_lmax} is required.")
            self.data = alike.load_data(
                ddir=self.ddir, variant=self.variant, lens_only=self.lens_only,
                like_corrections=not self.no_like_corrections,
                apply_hartlap=self.apply_hartlap, mock=self.mock,
                nsims_act=self.nsims_act, nsims_planck=self.nsims_planck,
                trim_lmax=self.trim_lmax, scale_cov=self.scale_cov,
                version=self.version, act_cmb_rescale=self.act_cmb_rescale,
                act_calib=self.act_calib)
            self.requested_cls = (["pp"] if self.no_like_corrections
                                  else ["tt", "te", "ee", "bb", "pp"])
        self.log.info(
            f"ACT DR6 lensing: variant={self.variant} lens_only={self.lens_only} "
            f"like_corrections={not self.no_like_corrections} "
            f"growth_ratio={'ON' if self.apply_growth_ratio else 'off'}")

    def get_requirements(self):
        req = super().get_requirements()
        if self.apply_growth_ratio:
            req.update({p: None for p in ["ombh2", "omch2", "H0", "ns", "As"]})
            if self.growth_model == "gamma":
                req["gamma_growth"] = None
            elif self.growth_model == "detg":
                req["beta_detg"] = None
        if self.varying_cmb_alens:
            req["Alens"] = None
        return req


    def loglike(self, cl, **params_values):
        if not self.apply_growth_ratio:
            return super().loglike(cl, **params_values)

        ell = cl["ell"]
        Alens = self.provider.get_param("Alens") if self.varying_cmb_alens else 1.0
        clpp = cl["pp"] / Alens
        cl_kk = (self.get_limber_clkk(**params_values) if self.limber
                 else alike.pp_to_kk(clpp, ell))

        built = CCLState.build_ccl(params_values, growth_model=self.growth_model,
                          p_detg=self.p_detg, m_nu=self.m_nu,
                          mass_split=self.mass_split)
        if built is None:                       # beta >= 1 之類的不合法點
            return -np.inf
        st, k_arr, a_nl = built
        r = CCLState.kk_transfer_ratio(st, k_arr, a_nl, lmax=self.lmax)
        # ----------------------------------------------------------

        if r.size < ell.size:                   # pad with the last value
            r = np.concatenate([r, np.full(ell.size - r.size, r[-1])])
        cl_kk = cl_kk * r[:ell.size]

        z = np.zeros_like(ell, dtype=float)
        tt, ee, te, bb = (cl.get(k, z) for k in ("tt", "ee", "te", "bb"))

        logp = alike.generic_lnlike(
            self.data, ell, cl_kk, ell, tt, ee, te, bb,
            self.trim_lmax, do_norm_corr=not self.act_cmb_rescale,
            act_calib=self.act_calib,
            no_actlike_cmb_corrections=self.no_actlike_cmb_corrections)
        self.log.debug(f"ACT-DR6-lensing lnLike = {logp} (chi2 = {-2 * logp})")
        return logp