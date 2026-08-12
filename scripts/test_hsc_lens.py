# Integrate in real space

import os
import numpy as np
from cobaya.likelihood import Likelihood
from astropy.io import fits
import pyccl as ccl

import sys
sys.path.insert(0, '/home/weichen/cosmo_practice/chains/actxdesi_new/actxdesi_new')
try:
    from growth_model import R_of_z, f_ratio_of_z
except ImportError:
    R_of_z = f_ratio_of_z = None

om_m_ref = 0.279
C_OVER_H0 = 2997.92458
alpha_mag = [2.259, 3.563, 3.729]
SIGC_CONST = 1.6624e18

def E_flat(z, Om):
    return np.sqrt(Om*(1.0+z)**3 + (1.0-Om))

def chi_hinv_flat(z, Om, n=512):
    zz = np.linspace(0.0, z, n)
    return C_OVER_H0 * np.trapezoid(1.0/E_flat(zz, Om), zz)

def sigcrit_inv_mean(zl, om, z_s, nz_s):
    Dl = chi_hinv_flat(zl, om)
    Ds = np.array([chi_hinv_flat(z, om) for z in z_s])
    integ = np.where(z_s > zl, Dl * (Ds - Dl) / np.where(Ds > 0, Ds, 1.0), 0.0) * nz_s
    return np.trapezoid(integ, z_s) / np.trapezoid(nz_s, z_s)


class HSC_Lens_Growth(Likelihood): 
    # derived paras
    output_params = ["S8_z_L1", "S8_z_L2", "S8_z_L3"]
    data_folder: str = ""
    dataset_file: str = "dataset.fits"

    use_growth: bool = True

    def initialize(self):
        self.data_path = os.path.join(self.data_folder, self.dataset_file)
        self.log.info(f"Loading HSC 3x2pt Lens data: {self.data_path}")
        
        with fits.open(self.data_path) as hdul:
            self.ds_table  = hdul['ds'].data
            self.wp_table  = hdul['wp'].data
            self.xip_table = hdul['xip'].data
            self.xim_table = hdul['xim'].data

            # self.log.info(f"ds ANG = {self.ds_table['ANG']}")
            # self.log.info(f"ds VALUE = {self.ds_table['VALUE']}")

            # scale cut (Sugiyama et al. 2023)
            ds_rmax = {1: 30.0, 2: 40.0, 3: 80.0}

            self.ds_cut = np.zeros(len(self.ds_table), dtype=bool)
            for bin_id, rmax in ds_rmax.items():
                self.ds_cut |= ((self.ds_table['BIN1'] == bin_id)  # |= (OR assignment)
                               & (self.ds_table['ANG'] >= 12.0) 
                               & (self.ds_table['ANG'] <= rmax))

            self.wp_cut = ((self.wp_table['ANG'] >= 8.0) 
                           & (self.wp_table['ANG'] <= 80.0))
            self.xip_cut = ((self.xip_table['ANG'] >= 7.9)
                           & (self.xip_table['ANG'] <= 50.1))

            self.xim_cut = ((self.xim_table['ANG'] >= 31.6)
                           & (self.xim_table['ANG'] <= 158.0)
            )

            # data vector
            self.data_vector = np.concatenate([
                self.ds_table['VALUE'][self.ds_cut],
                self.xip_table['VALUE'][self.xip_cut],
                self.xim_table['VALUE'][self.xim_cut],
                self.wp_table['VALUE'][self.wp_cut]
            ])

            n_ds_full = len(self.ds_table)
            n_xip_full = len(self.xip_table)
            n_xim_full = len(self.xim_table)

            off_xip = n_ds_full
            off_xim = n_ds_full + n_xip_full
            off_wp = n_ds_full + n_xip_full + n_xim_full

            keep_idx = np.concatenate([
                np.where(self.ds_cut)[0],
                np.where(self.xip_cut)[0] + off_xip,
                np.where(self.xim_cut)[0] + off_xim,
                np.where(self.wp_cut)[0] + off_wp
            ])

            full_cov = hdul['COVMAT'].data
            cut_cov = full_cov[np.ix_(keep_idx, keep_idx)]
            self.inv_cov = (np.linalg.inv(cut_cov) * (107*13 - len(self.data_vector) - 2) / (107*13 - 1))  # Hartlap correction
            if self.inv_cov.shape != (len(self.data_vector), len(self.data_vector)):
                raise ValueError("Scale-cut mismatch: "
                    f"data vector has length {len(self.data_vector)}, "
                    f"but inverse covariance has shape {self.inv_cov.shape}")
            
            self.log.info("After scale cut: "
                        f"ds={self.ds_cut.sum()}\n"
                        f"xip={self.xip_cut.sum()}\n"
                        f"xim={self.xim_cut.sum()}\n"
                        f"wp={self.wp_cut.sum()}\n"
                        f"total={len(self.data_vector)}")

            # redshift
            self.z_s = hdul['nz_source'].data['Z_MID']
            self.nz_s = hdul['nz_source'].data['BIN1']
            self.z_l = hdul['nz_lens'].data['Z_MID']
            self.nz_l_list = [hdul['nz_lens'].data[f'BIN{i+1}'] for i in range(3)]

            if self.use_growth and R_of_z is None:
                raise ImportError("use_growth=True 但 growth_model 匯入失敗,請檢查 python_path")

    def get_requirements(self): 
        req = {
            'sigma8': None, 'H0': None, 'ombh2': None, 'omch2': None, 'ns': None,
            'b1': None, 'b2': None, 'b3': None,
            'AIA': None, 'dm_0': None, 'dpz_0': None
        }
        if self.use_growth:
            req['growth_ratio'] = None
        return req

    @staticmethod
    def _growth_pk2d(cosmo, gr):
        """ξ± 與 magnification 的 lensing kernel 橫跨 z,R(z) 必須逐 z 加權,
        不能像 ΔΣ/w_p 那樣乘一個常數"""
        if gr is None:
            return None
        z = np.linspace(0.0, 5.0, 120)
        a = 1.0 / (1.0 + z)
        k = np.logspace(-4, 1.5, 300)   # 1/Mpc
        R = R_of_z(gr, z)
        pk = np.array([ccl.nonlin_matter_power(cosmo, k, ai) for ai in a])
        return ccl.Pk2D(a_arr=a[::-1],  # CCL 要求 a 遞增
                        lk_arr=np.log(k), pk_arr=np.log((pk * R[:, None]**2)[::-1]), is_logp=True)

    def get_theory_prediction(self, cosmo, gr=None, **params):
        b = [params[f'b{i+1}'] for i in range(3)]
        A_IA = params['AIA']
        dm = params['dm_0']
        dzph = params['dpz_0']
 
        h = cosmo['h']
        om = cosmo['Omega_c'] + cosmo['Omega_b']
 
        z_l_eff = [0.2607, 0.5106, 0.6264]
        pi_max = 100.0
        ell = np.geomspace(0.1, 1e5, 1024)
 
        pk2d = self._growth_pk2d(cosmo, gr)

        m_ds, m_wp = [], []
        rho_m_0 = ccl.rho_x(cosmo, 1.0, 'matter', is_comoving=True)  # M☉/Mpc³
        nz_true = np.interp(self.z_s + dzph, self.z_s, self.nz_s, left=0.0, right=0.0)   # Eq. (18)

        for i in range(3):
            zl = z_l_eff[i]
            a_l = 1.0 / (1.0 + z_l_eff[i])

            R_l = R_of_z(gr, zl) if gr is not None else 1.0
            fr = f_ratio_of_z(gr, zl) if gr is not None else 1.0

            E_C, E_ref = E_flat(zl, om), E_flat(zl, om_m_ref)
            chi_C, chi_ref = chi_hinv_flat(zl, om), chi_hinv_flat(zl, om_m_ref)

            f_ds = (sigcrit_inv_mean(zl, om, self.z_s, nz_true) / sigcrit_inv_mean(zl, om_m_ref, self.z_s, self.nz_s))

            R_fac, E_fac = chi_C / chi_ref, E_C / E_ref
            pimax_wp = (E_ref / E_C) * pi_max

            mask_ds = (self.ds_table['BIN1'] == (i + 1)) & self.ds_cut
            mask_wp = (self.wp_table['BIN1'] == (i + 1)) & self.wp_cut
            rp_ds = self.ds_table['ANG'][mask_ds]   # Mpc/h
            rp_wp = self.wp_table['ANG'][mask_wp]   # Mpc/h
            rp_ds_true = R_fac * rp_ds
            rp_wp_true = R_fac * rp_wp

            # Magnification contribution to ΔΣ
            kern_Mpc = sigcrit_inv_mean(zl, om, self.z_s, nz_true) / h
            Sig_c = SIGC_CONST / ((1.0 + zl) * kern_Mpc)

            tr_l = ccl.WeakLensingTracer(cosmo, dndz=(self.z_l, self.nz_l_list[i]))
            tr_s = ccl.WeakLensingTracer(cosmo, dndz=(self.z_s, nz_true))

            ell_mag = np.geomspace(0.1, 1e5, 512)
            cl_ls = ccl.angular_cl(cosmo, tr_l, tr_s, ell=ell_mag, p_of_k_a=pk2d)
            theta_deg = (rp_ds_true / chi_C) * (180.0 / np.pi)

            gt = ccl.correlation(
                cosmo, ell=ell_mag, C_ell=cl_ls,
                theta=theta_deg, type='NG', method='FFTLog'
            )

            ds_mag = (2.0 * (alpha_mag[i] - 1.0) * Sig_c * gt / (h * 1e12))

            pi_ds = np.linspace(0.0, pi_max, 300)     # Mpc/h
            pi_wp = np.linspace(0.0, pimax_wp, 300)   # wp turn to Πmax
            # self.log.info(
            #     f"Bin {i+1}:\n "
            #     f"ds points={len(rp_ds_true)}\n "
            #     f"wp points={len(rp_wp_true)}\n"
            #     f"rp_ds={rp_ds}\n"
            #     f"rp_wp={rp_wp}"\n)
            
            rmax_all = max(
                np.max(self.ds_table['ANG']),
                np.max(self.wp_table['ANG'])
            )
            r_grid = np.geomspace(1e-3, rmax_all*20 + 100, 1000)   # Mpc/h

            # for CCL: 傳入 r_grid/h (Mpc)
            xi_mm = ccl.correlation_3d(cosmo, a=a_l, r=r_grid/h)
            xi_gm_grid = b[i] * R_l**2 * xi_mm
            xi_gg_grid = b[i]**2 * R_l**2 * xi_mm

            f_growth = ccl.growth_rate(cosmo, a_l)
            beta = f_growth * fr / b[i]

            def ds_at_rp(rp, xi_gm_grid=xi_gm_grid, r_grid=r_grid, pi_arr=pi_ds, rho_m=rho_m_0, h=h):
                # Σ(rp)
                r3d = np.sqrt(rp**2 + pi_arr**2)
                xi_pi = np.interp(r3d, r_grid, xi_gm_grid)
                Sigma = 2 * rho_m * np.trapezoid(xi_pi, pi_arr) / (h**2 * 1e12)

                # Σ(rp') 在 rp_inner 格點上
                rp_inner = np.linspace(1e-3, rp, 100)
                r3d_inner = np.sqrt(rp_inner[None,:]**2 + pi_arr[:,None]**2)
                xi_inner = np.interp(r3d_inner, r_grid, xi_gm_grid)
                Sigma_inner = 2 * rho_m * np.trapezoid(xi_inner, pi_arr, axis=0) / (h**2 * 1e12)

                # bar_Σ(<rp)
                bar_Sigma = 2 / rp**2 * np.trapezoid(Sigma_inner * rp_inner, rp_inner)

                return bar_Sigma - Sigma
            
            m_ds.extend([(1.0 + dm) * f_ds * (ds_at_rp(rp) + dsm) for rp, dsm in zip(rp_ds_true, ds_mag)])

            from scipy.integrate import cumulative_trapezoid
            J3_grid = cumulative_trapezoid(xi_gg_grid * r_grid**2, r_grid, initial=0) / r_grid**3
            J5_grid = cumulative_trapezoid(xi_gg_grid * r_grid**4, r_grid, initial=0) / r_grid**5

            def wp_at_rp(rp, xi_gg_grid=xi_gg_grid, r_grid=r_grid, pi_arr=pi_wp, beta=beta, J3_grid=J3_grid, J5_grid=J5_grid):
                r3d = np.sqrt(rp**2 + pi_arr**2)
                mu = pi_arr / r3d
                
                # (r3d, mu) 
                xi_lin = np.interp(r3d, r_grid, xi_gg_grid)
                J3 = np.interp(r3d, r_grid, J3_grid)
                J5 = np.interp(r3d, r_grid, J5_grid)
                
                xi0 = (1 + (2/3)*beta + (1/5)*beta**2) * xi_lin
                xi2 = ((4/3)*beta + (4/7)*beta**2) * (xi_lin - 3*J3)
                xi4 = (8/35)*beta**2 * (xi_lin + (15/2)*J3 - (35/2)*J5)

                # Legendre polynomials
                L2 = 0.5 * (3*mu**2 - 1)
                L4 = 0.125 * (35*mu**4 - 30*mu**2 + 3)
                
                # xi^s(r, mu)
                xi_s = xi0 + xi2*L2 + xi4*L4

                return 2 * np.trapezoid(xi_s, pi_arr)   # Mpc/h

            m_wp.extend([E_fac * wp_at_rp(rp) for rp in rp_wp_true])
            # self.log.info(f"Bin {i+1} timing: ds={t_ds:.2f}s, J3J5={t_j:.2f}s, wp={t_wp:.2f}s")

        ia = (self.z_s, np.full_like(self.z_s, float(A_IA)))
        s_tracer = ccl.WeakLensingTracer(cosmo, dndz=(self.z_s, nz_true), ia_bias=ia)

        cl_ss = ccl.angular_cl(cosmo, s_tracer, s_tracer, ell=ell, p_of_k_a=pk2d)
        fac_m = (1.0 + dm)**2

        xip = fac_m * ccl.correlation(
            cosmo, ell=ell, C_ell=cl_ss,
            theta=self.xip_table['ANG'][self.xip_cut]/60.,
            type='GG+', method='FFTLog'
        )
        xim = fac_m * ccl.correlation(
            cosmo, ell=ell, C_ell=cl_ss,
            theta=self.xim_table['ANG'][self.xim_cut]/60.,
            type='GG-', method='FFTLog'
        )

        return np.concatenate([m_ds, np.atleast_1d(xip), np.atleast_1d(xim), m_wp])

    def logp(self, _derived=None, **params_values):
        H0, ombh2, omch2=params_values['H0'], params_values['ombh2'], params_values['omch2']
        h = H0 / 100.0
        s8_planck = self.provider.get_param('sigma8')
        if s8_planck is None: return -np.inf
        
        cosmo = ccl.Cosmology(
            Omega_c=float(omch2/h**2), Omega_b=float(ombh2/h**2), h=h, 
            sigma8=float(s8_planck), n_s=params_values['ns'], matter_power_spectrum='halofit'
        )   # origin sigma8

        gr = self.provider.get_growth_ratio() if self.use_growth else None
        m = self.get_theory_prediction(cosmo, gr=gr, **params_values)
        
        if len(m) != len(self.data_vector):
            raise ValueError("Theory/data length mismatch: "
                f"theory={len(m)}, data={len(self.data_vector)}. "
                "Check scale cuts in get_theory_prediction().")
        
        diff = self.data_vector - m
        chi2 = diff @ self.inv_cov @ diff

        if _derived is not None:
            om_m = (ombh2 + omch2) / h**2
            _derived["sigma8"] = s8_planck
            z_lens = [0.2607, 0.5106, 0.6264]
            for i, z in enumerate(z_lens):
                growth = ccl.growth_factor(cosmo, 1.0/(1.0+z))
                R_l = R_of_z(gr, z) if gr is not None else 1.0
                _derived[f"S8_z_L{i+1}"] = (s8_planck * R_l * growth * np.sqrt(om_m / 0.3))

        return -0.5 * chi2