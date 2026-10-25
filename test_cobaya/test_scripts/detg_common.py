"""共用工具：讀 DETG chain（Planck + HSC 3x2pt + ACT DR6 lensing）並加上分析用的 derived 量。

chain 中各參數的意義（依 detg_hsc_act.yaml / hsc_lens.py）
-----------------------------------------------------------------
beta_detg        : DETG 的 beta；p 固定在 yaml（每條 chain 一個 p），本模組從 <root>.updated.yaml 自動讀取
sigma8, S8       : 由 HSC_Lens 輸出，**已經是 DETG 值**：sigma8 = sigma8_LCDM * sqrt(alpha(z=0)) = sigma8_LCDM * sqrt(1-beta)
S8_z_L1..3       : HSC 輸出，sigma8_LCDM * sqrt(alpha(z_l)) * D_LCDM(z_l) * sqrt(Om/0.3)
om_m             : yaml 中的 lambda derived（含 0.06 eV 中微子）
Alens, A_planck, tau : Planck 相關

本模組加上的 derived（addDerived）
-----------------------------------------------------------------
sigma8_lcdm      : sigma8 / sqrt(alpha(0))  —— CMB 正規化的 LCDM sigma8
S8_lcdm          : sigma8_lcdm * sqrt(om_m/0.3)
f_ratio_z005     : f_DETG / f_LCDM 在 z = 0.05（< 0 表示 z->0 時成長率為負，見 plot_result.py 第 5 節）
"""

import os
import sys
import numpy as np
import yaml
from getdist import loadMCSamples

# ------------------------------------------------------------------ 路徑設定（依需要修改）
CHAIN_BASE = "/home/andromeda/cosmo_practice/chains/"          # cobaya run 的 output 相對於 ~/cosmo_practice
THEORY_DIR = "/home/andromeda/cosmo_practice/test_cobaya/theory"
OUTDIR     = "./detg_plots"
BURN       = 0.3

# ------------------------------------------------------------------ 參考值
# Terasawa et al. 2025 (arXiv:2505.09176) Table II：Planck + ACT DR6 lensing + DESI Y1 BAO + HSC Y3 cosmic shear
# (median, +err, -err)；資料組合與本分析不同（他們有 BAO、HSC 只用 cosmic shear、nuisance 固定），僅供對照
TERASAWA_BETA = {1: (0.074, 0.067, 0.078), 2: (0.152, 0.111, 0.094), 3: (0.251, 0.151, 0.123),
                 4: (0.367, 0.168, 0.176), 5: (0.443, 0.227, 0.203), 6: (0.578, 0.260, 0.223)}

# detg_act_hsc_joint_check.ipynb（real 模式）的 Fisher 相關係數 —— **只有 ACT + HSC、沒有 Planck**
# 加入 Planck 後 A_s、背景會被釘住，相關係數應明顯變小；僅作「方向」的對照
FISHER_RHO_ACTHSC = {1: {"logA": +0.655, "omch2": -0.292, "H0": -0.273},
                     3: {"logA": +0.440, "omch2": -0.278, "H0": -0.189},
                     5: {"logA": +0.347, "omch2": -0.233, "H0": -0.137}}

PLANCK_S8, PLANCK_S8ERR = 0.832, 0.013          # Planck 2018 TT,TE,EE+lowE（LCDM）
HSC_Z   = np.array([0.2607, 0.5106, 0.6264])     # HSC lens bins（z_l_eff，與 hsc_lens.py 相同）
HSC_LBL = ["LOWZ", "CMASS1", "CMASS2"]


# ------------------------------------------------------------------ DETG（向量化；與 growth_model.py 相同公式）
def x_de(z, om):
    """Omega_DE(z)/Omega_DE(0) = 1/E^2(z)（平坦 LCDM 背景，與 GrowthModel._x 相同）"""
    z = np.asarray(z, dtype=float)
    return 1.0 / (om * (1.0 + z) ** 3 + (1.0 - om))


def omega_m_z(z, om):
    z = np.asarray(z, dtype=float)
    return om * (1.0 + z) ** 3 * x_de(z, om)


def alpha(z, om, beta, p):
    """alpha(z) = P_L/P_L^LCDM = 1 - beta x^p；p=None 代表 LCDM（alpha = 1）"""
    if p is None:
        return np.ones(np.broadcast(np.asarray(z), np.asarray(om)).shape)
    return 1.0 - beta * x_de(z, om) ** p


def f_lcdm_approx(z, om):
    """f_LCDM ≈ Omega_m(z)^0.55（誤差 ~0.5%，只用於判斷 f_DETG 的符號與大小）"""
    return omega_m_z(z, om) ** 0.55


def f_detg(z, om, beta, p):
    """f = f_LCDM + 0.5 dln(alpha)/dln(a) = f_LCDM - 1.5 beta p Omega_m(a) x^p / alpha（GrowthModel.f）"""
    fl = f_lcdm_approx(z, om)
    if p is None:
        return fl
    x = x_de(z, om)
    return fl - 1.5 * beta * p * omega_m_z(z, om) * x ** p / alpha(z, om, beta, p)


def D_lcdm(z, om, n=2000):
    """LCDM 線性成長因子，D(z=0)=1（積分式，忽略輻射與中微子；與 CCL 差 ~0.3%）"""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    om = np.atleast_1d(np.asarray(om, dtype=float))

    def unnorm(zz, o):
        a_hi = 1.0 / (1.0 + zz)
        a = np.linspace(1e-5, 1.0, n)
        E = np.sqrt(o / a ** 3 + 1.0 - o)
        integ = 1.0 / (a * E) ** 3
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(a))])
        Ehi = np.sqrt(o / a_hi ** 3 + 1.0 - o)
        return 2.5 * o * Ehi * np.interp(a_hi, a, cum)

    out = np.array([[unnorm(zz, o) / unnorm(0.0, o) for zz in z] for o in om])
    return out.squeeze()


def check_against_growthmodel(om=0.31, beta=0.2, p=3):
    """確認向量化公式與 growth_model.GrowthModel 一致（載入時自動執行一次）"""
    if THEORY_DIR not in sys.path:
        sys.path.insert(0, THEORY_DIR)
    try:
        from growth_model import GrowthModel
    except ImportError:
        print(f"[detg_common] 找不到 growth_model.py（{THEORY_DIR}），略過一致性檢查")
        return
    zz = np.array([0.0, 0.3, 1.0])
    gr = GrowthModel("detg", om, par=beta, p=p)
    a_ref = gr.alpha(1.0 / (1.0 + zz))
    f_ref = gr.f(zz, f_lcdm_approx(zz, om))
    assert np.allclose(alpha(zz, om, beta, p), a_ref, rtol=1e-12), "alpha 與 GrowthModel 不一致"
    assert np.allclose(f_detg(zz, om, beta, p), f_ref, rtol=1e-12), "f 與 GrowthModel 不一致"


# ------------------------------------------------------------------ chain 讀取
def root_path(root):
    return root if os.path.isabs(root) else os.path.join(CHAIN_BASE, root)


def exists(root):
    rp = root_path(root)
    return os.path.exists(rp + ".updated.yaml") or os.path.exists(rp + ".1.txt") or os.path.exists(rp + ".txt")


def read_info(root):
    fn = root_path(root) + ".updated.yaml"
    if not os.path.exists(fn):
        fn = root_path(root) + ".input.yaml"
    with open(fn) as f:
        return yaml.safe_load(f)


def growth_setup(info):
    """從 likelihood 區塊讀 growth_model 與 p_detg；ACT 與 HSC 必須一致"""
    found = {}
    for name, opts in (info.get("likelihood") or {}).items():
        if isinstance(opts, dict) and "growth_model" in opts:
            found[name] = (opts.get("growth_model"), opts.get("p_detg"))
    if not found:
        return "none", None, found
    models = {v[0] for v in found.values()}
    ps = {v[1] for v in found.values() if v[0] == "detg"}
    if len(models) > 1 or len(ps) > 1:
        print(f"!! likelihood 之間 growth_model / p_detg 不一致：{found}")
    model = models.pop()
    p = float(ps.pop()) if (model == "detg" and ps) else None
    return model, p, found


def prior_info(info, name):
    """回傳 ('flat', lo, hi) / ('norm', loc, scale) / None（固定或不存在）"""
    v = (info.get("params") or {}).get(name)
    if not isinstance(v, dict) or "prior" not in v:
        return None
    pr = v["prior"]
    if "min" in pr and "max" in pr:
        return ("flat", float(pr["min"]), float(pr["max"]))
    if pr.get("dist") == "norm":
        return ("norm", float(pr["loc"]), float(pr["scale"]))
    return None


def col(s, name):
    j = s.paramNames.numberOfName(name)
    return s.samples[:, j] if j >= 0 else None


def has(s, name):
    return s.paramNames.numberOfName(name) >= 0


def stat(s, name):
    x = col(s, name)
    if x is None:
        return None
    w = s.weights
    m = np.average(x, weights=w)
    return m, np.sqrt(np.average((x - m) ** 2, weights=w))


def corr(s, a, b):
    xa, xb = col(s, a), col(s, b)
    if xa is None or xb is None:
        return np.nan
    w = s.weights
    ma, mb = np.average(xa, weights=w), np.average(xb, weights=w)
    cab = np.average((xa - ma) * (xb - mb), weights=w)
    return cab / np.sqrt(np.average((xa - ma) ** 2, weights=w) * np.average((xb - mb) ** 2, weights=w))


def load(root, burn=BURN, label=None):
    """讀 chain，加上 sigma8_lcdm、S8_lcdm、f_ratio_z005；回傳 (samples, meta)"""
    s = loadMCSamples(root_path(root), settings={"ignore_rows": burn})
    info = read_info(root)
    model, p, found = growth_setup(info)
    meta = dict(root=root, label=label or root, model=model, p=p, info=info, growth_opts=found)

    om = col(s, "om_m")
    if om is None:
        om = col(s, "omegam")
    beta = col(s, "beta_detg")
    if beta is None or model != "detg":
        beta = np.zeros(s.samples.shape[0])
    meta["has_beta"] = has(s, "beta_detg") and model == "detg"

    s8 = col(s, "sigma8")
    if s8 is not None and om is not None:
        a0 = alpha(0.0, om, beta, p)
        if not has(s, "sigma8_lcdm"):
            s.addDerived(s8 / np.sqrt(a0), name="sigma8_lcdm", label=r"\sigma_8^{\Lambda{\rm CDM}}")
        if not has(s, "S8_lcdm"):
            s.addDerived(s8 / np.sqrt(a0) * np.sqrt(om / 0.3), name="S8_lcdm", label=r"S_8^{\Lambda{\rm CDM}}")
    if om is not None and meta["has_beta"] and not has(s, "f_ratio_z005"):
        s.addDerived(f_detg(0.05, om, beta, p) / f_lcdm_approx(0.05, om),
                     name="f_ratio_z005", label=r"f/f_{\Lambda{\rm CDM}}(z{=}0.05)")
    return s, meta


def convergence(root, burn=BURN, params=("beta_detg", "S8", "Alens", "logA")):
    """收斂檢查：讀 cobaya 的 .progress（若有），並由各 chain 檔計算單變數 Gelman-Rubin R-1"""
    import glob
    rp = root_path(root)
    out = {}
    if os.path.exists(rp + ".progress"):
        try:
            prog = np.atleast_2d(np.loadtxt(rp + ".progress"))
            with open(rp + ".progress") as f:
                hdr = f.readline().lstrip("#").split()
            last = dict(zip(hdr, prog[-1]))
            out["progress"] = {k: last[k] for k in ("N", "Rminus1", "Rminus1_cl") if k in last}
        except Exception:
            pass
    files = [f for f in sorted(glob.glob(rp + ".[0-9]*.txt"))]
    chains, names = [], None
    for f in files:
        with open(f) as fh:
            first = fh.readline()
        if not first.startswith("#"):
            continue
        names = first.lstrip("#").split()
        d = np.atleast_2d(np.loadtxt(f))
        chains.append(d[int(burn * len(d)):])
    if len(chains) >= 2 and names:
        for prm in params:
            if prm not in names:
                continue
            j = names.index(prm)
            m, v, n = [], [], []
            for d in chains:
                w, x = d[:, 0], d[:, j]
                mu = np.average(x, weights=w)
                m.append(mu); v.append(np.average((x - mu) ** 2, weights=w)); n.append(w.sum())
            m, v = np.array(m), np.array(v)
            out[prm] = float(np.var(m, ddof=1) / np.mean(v))      # (between-chain var of means)/(mean within var)
    return out


def describe(meta):
    tag = f"p={meta['p']:g}" if meta["p"] is not None else "LCDM (growth_model=none)"
    return f"{meta['label']}  [{tag}]"


os.makedirs(OUTDIR, exist_ok=True)
check_against_growthmodel()
