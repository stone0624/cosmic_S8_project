# Integrate in real space

import os
import numpy as np
from cobaya.likelihood import Likelihood
from astropy.io import fits
import pyccl as ccl

om_m_ref = 0.279   # reference Omega_m used for coordinate rescaling
 
def E_flat(z, om):
    return np.sqrt(om * (1 + z)**3 + (1 - om))
 
def chi_hinv_flat(z, om, n=512):
    zz = np.linspace(0, z, n)
    return 2997.925 * np.trapezoid(1.0 / E_flat(zz, om), zz)

class HSC_Lens(Likelihood): 
    # derived paras
    output_params = ["S8_z_L1", "S8_z_L2", "S8_z_L3"]
    data_folder: str = ""
    dataset_file: str = "dataset.fits"

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
            self.ds_cut = self.ds_table['ANG'] >= 12.0   # Mpc/h
            self.wp_cut = self.wp_table['ANG'] >= 8.0    # Mpc/h

            # data vector
            self.data_vector = np.concatenate([
                self.ds_table['VALUE'][self.ds_cut],
                self.xip_table['VALUE'],
                self.xim_table['VALUE'],
                self.wp_table['VALUE'][self.wp_cut]
            ])

            n_ds_full = len(self.ds_table)
            n_xip = len(self.xip_table)
            n_xim = len(self.xim_table)

            ds_idx = np.where(self.ds_cut)[0]
            xip_idx = np.arange(n_ds_full, n_ds_full + n_xip)
            xim_idx = np.arange(n_ds_full + n_xip, n_ds_full + n_xip + n_xim)
            wp_idx = np.where(self.wp_cut)[0] + n_ds_full + n_xip + n_xim

            keep_idx = np.concatenate([ds_idx, xip_idx, xim_idx, wp_idx])
            full_cov = hdul['COVMAT'].data
            self.inv_cov = np.linalg.inv(full_cov[np.ix_(keep_idx, keep_idx)])

            self.log.info(f"After scale cut: ds={self.ds_cut.sum()}, wp={self.wp_cut.sum()}, "
                        f"xip={n_xip}, xim={n_xim}, total={len(self.data_vector)}")

            # redshift
            self.z_s = hdul['nz_source'].data['Z_MID']
            self.nz_s = hdul['nz_source'].data['BIN1']
            self.z_l = hdul['nz_lens'].data['Z_MID']
            self.nz_l_list = [hdul['nz_lens'].data[f'BIN{i+1}'] for i in range(3)]

    def get_requirements(self): 
        return {
            'sigma8': None, 'H0': None, 'ombh2': None, 'omch2': None, 'ns': None,
            'X1': None, 'X2': None, 'X3': None,
            'b1': None, 'b2': None, 'b3': None
        }

    def get_theory_prediction(self, cosmo, **params):
        X = [params[f'X{i+1}'] for i in range(3)]
        b = [params[f'b{i+1}'] for i in range(3)]
        h = cosmo['h']
        om = cosmo['Omega_m']

        z_l_eff = [0.2607, 0.5106, 0.6264]
        pi_max = 100.0
        ell = np.geomspace(2, 50000, 500)
        s_tracer = ccl.WeakLensingTracer(cosmo, dndz=(self.z_s, self.nz_s))

        m_ds, m_wp = [], []
        rho_m_0 = ccl.rho_x(cosmo, 1.0, 'matter', is_comoving=True)  # M☉/Mpc³

        for i in range(3):
            zl = z_l_eff[i]
            a_l = 1.0 / (1.0 + z_l_eff[i])

            E_C,  E_ref = E_flat(zl, om), E_flat(zl, om_m_ref)
            chi_C, chi_ref = chi_hinv_flat(zl, om), chi_hinv_flat(zl, om_m_ref)
            R_fac, E_fac = chi_C / chi_ref, E_C / E_ref
            pimax_wp = (E_ref / E_C) * pi_max

            mask_ds = (self.ds_table['BIN1'] == (i + 1)) & self.ds_cut
            mask_wp = (self.wp_table['BIN1'] == (i + 1)) & self.wp_cut
            rp_ds = self.ds_table['ANG'][mask_ds]   # Mpc/h
            rp_wp = self.wp_table['ANG'][mask_wp]   # Mpc/h
            rp_ds_true = R_fac * rp_ds
            rp_wp_true = R_fac * rp_wp
            
            pi_ds = np.linspace(1e-4, pi_max, 300)     # Mpc/h
            pi_wp = np.linspace(1e-4, pimax_wp, 300)   # wp turn to Πmax

            r_max = max(rp_ds.max(), rp_wp.max()) * 20
            r_grid = np.geomspace(1e-3, r_max + 100, 1000)  # Mpc/h

            # for CCL: 傳入 r_grid/h (Mpc)
            xi_mm = ccl.correlation_3d(cosmo, a=a_l, r=r_grid/h)
            xi_gm_grid = b[i] * X[i]**2 * xi_mm
            xi_gg_grid = b[i]**2 * X[i]**2 * xi_mm

            f_growth = ccl.growth_rate(cosmo, a_l)
            beta = f_growth / b[i]

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
            
            m_ds.extend([E_fac * ds_at_rp(rp) for rp in rp_ds_true])

            J3_grid = np.zeros(len(r_grid))
            J5_grid = np.zeros(len(r_grid))
            for j, r in enumerate(r_grid):
                mask = r_grid <= r
                if mask.sum() < 2:
                    continue
                J3_grid[j] = np.trapezoid(xi_gg_grid[mask] * r_grid[mask]**2, r_grid[mask]) / r**3
                J5_grid[j] = np.trapezoid(xi_gg_grid[mask] * r_grid[mask]**4, r_grid[mask]) / r**5

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

            m_wp.extend([wp_at_rp(rp) for rp in rp_wp_true])
            # self.log.info(f"Bin {i+1} timing: ds={t_ds:.2f}s, J3J5={t_j:.2f}s, wp={t_wp:.2f}s")

        cl_ss = ccl.angular_cl(cosmo, s_tracer, s_tracer, ell=ell)
        xip = ccl.correlation(cosmo, ell=ell, C_ell=cl_ss,
                            theta=self.xip_table['ANG']/60., type='GG+', method='FFTLog')
        xim = ccl.correlation(cosmo, ell=ell, C_ell=cl_ss,
                            theta=self.xim_table['ANG']/60., type='GG-', method='FFTLog')

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
        
        m = self.get_theory_prediction(cosmo, **params_values)

        # # --- debug ---
        # n_ds = self.ds_cut.sum()
        # n_xip = len(self.xip_table)
        # n_xim = len(self.xim_table)

        # self.log.info(f"theory ds[:3] = {m[:3]}")
        # self.log.info(f"data ds[:3] = {self.data_vector[:3]}")
        # self.log.info(f"ratio ds[:3] = {self.data_vector[:3] / m[:3]}")

        # # 各 bin 的 ratio（scale cut）
        # d_ds_all = self.data_vector[:n_ds]
        # m_ds_all = m[:n_ds]

        # ptr = 0
        # for i in range(3):
        #     mask_bin = (self.ds_table['BIN1'] == (i + 1)) & self.ds_cut
        #     n_bin = mask_bin.sum()
        #     if n_bin == 0:
        #         continue
        #     d_bin = d_ds_all[ptr:ptr + n_bin]
        #     m_bin = m_ds_all[ptr:ptr + n_bin]
        #     ratio = d_bin / m_bin
        #     self.log.info(f"Bin {i+1} (z={[0.2607,0.5106,0.6264][i]}) ds ratio mean={ratio.mean():.3f}, std={ratio.std():.3f}")
        #     self.log.info(f"ratio = {np.round(ratio, 3)}")
        #     ptr += n_bin

        # d_wp = self.data_vector[n_ds + n_xip + n_xim:]
        # m_wp = m[n_ds + n_xip + n_xim:]
        # self.log.info(f"wp ratio = {np.round(d_wp/m_wp, 3)}")
        # # --- debug Fin. ---

        diff = self.data_vector - m
        chi2 = diff @ self.inv_cov @ diff

        if _derived is not None:
            om_m = (ombh2 + omch2) / h**2
            _derived["sigma8"] = s8_planck
            z_lens = [0.2607, 0.5106, 0.6264]
            for i, z in enumerate(z_lens):
                growth = ccl.growth_factor(cosmo, 1.0/(1.0+z))
                _derived[f"S8_z_L{i+1}"] = s8_planck * params_values[f"X{i+1}"] * growth * np.sqrt(om_m/0.3)    # S8: 該層 Xi 修正後observe強度

        return -0.5 * chi2