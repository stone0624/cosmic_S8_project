"""S8x(z) 的後驗：DETG 下各紅移的結構振幅。

S8x(z) ≡ sigma8_LCDM (Om/0.3)^0.4 R(z),   R(z) = D_DETG(z)/D_LCDM(z) = sqrt(alpha(z))

- R 是逐 z 的比值（與 ACT×DESI LRG 的 S8x 定義相同：以 LCDM 成長外推回 z=0 的振幅）。
- chain 的 `sigma8` 是 HSC 輸出的 DETG 值，本程式用 sigma8_lcdm = sigma8 / sqrt(alpha(0))（detg_common 加上）。
- 每條 chain 的 p 自動從 <root>.updated.yaml 讀取；growth_model=none 的 chain 視為 LCDM（R ≡ 1）。
"""
import numpy as np
import matplotlib.pyplot as plt
from detg_common import (load, exists, col, alpha, D_lcdm, describe, OUTDIR,
                         PLANCK_S8, PLANCK_S8ERR, HSC_Z, HSC_LBL)

# ---------------------------------------------------------------- 設定（依實際 output 路徑修改；不存在的 root 會自動略過）
ROOTS = {
    r"DETG $p=3$, $A_L$ free": "detg_hsc_act/run",
    r"DETG $p=1$, $A_L$ free": "detg_p1_AL/run",
    r"DETG $p=5$, $A_L$ free": "detg_p5_AL/run",
    r"$\Lambda$CDM ($\beta=0$)": "lcdm_hsc_act/run",
}
COLORS = ["#D85A30", "#2a78d6", "#1baf7a", "#4B4B4B"]
NDRAW = 500
Z = np.linspace(0.0, 1.3, 120)
SEED = 1234
S8X_EXP = 0.4

# ACT×DESI LRG（Sailer+24, Zenodo 10.5281/zenodo.12613408, fiducial_noBAO_z{1,2,3,4}_pr4-dr6）
# 注意：這些點**不在本分析的 likelihood 中**，只作外部比較
ACT_Z   = np.array([0.47, 0.63, 0.79, 0.92])
ACT_S8X = np.array([0.7328, 0.7769, 0.7843, 0.7921])
ACT_ERR = np.array([0.0568, 0.0403, 0.0352, 0.0309])


def draw_curves(s, meta, n_draw=NDRAW):
    rng = np.random.default_rng(SEED)
    w = s.weights
    idx = rng.choice(len(w), size=n_draw, replace=True, p=w / w.sum())
    s8l, om = col(s, "sigma8_lcdm"), col(s, "om_m")
    if s8l is None or om is None:
        raise KeyError(f"{meta['root']}：chain 缺 sigma8 或 om_m")
    beta = col(s, "beta_detg") if meta["has_beta"] else np.zeros(len(w))
    om_i, b_i, s8_i = om[idx][:, None], beta[idx][:, None], s8l[idx][:, None]
    R = np.sqrt(alpha(Z[None, :], om_i, b_i, meta["p"]))
    return s8_i * (om_i / 0.3) ** S8X_EXP * R


# ---------------------------------------------------------------- 讀 chain
S, M, CUR = {}, {}, {}
for lab, root in ROOTS.items():
    if not exists(root):
        print(f"略過（找不到）：{root}")
        continue
    S[lab], M[lab] = load(root, label=lab)
    CUR[lab] = draw_curves(S[lab], M[lab])
    print("讀入", describe(M[lab]))
if not S:
    raise SystemExit("沒有任何 chain 可畫；請修改 ROOTS")

# ---------------------------------------------------------------- 畫圖
fig, ax = plt.subplots(figsize=(7.2, 5.0))
first = next(iter(S))
om_ref = np.average(col(S[first], "om_m"), weights=S[first].weights)
pl_conv = (om_ref / 0.3) ** (S8X_EXP - 0.5)            # S8 (指數 0.5) 換算到指數 0.4
ax.axhspan((PLANCK_S8 - PLANCK_S8ERR) * pl_conv, (PLANCK_S8 + PLANCK_S8ERR) * pl_conv,
           color="0.75", alpha=0.4, lw=0, zorder=0)
ax.axhline(PLANCK_S8 * pl_conv, color="0.45", lw=1.0, ls="--", zorder=1,
           label=r"Planck 2018 $S_8$ ($\Lambda$CDM, converted)")

for (lab, cur), color in zip(CUR.items(), COLORS):
    lo95, lo68, med, hi68, hi95 = np.percentile(cur, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(Z, lo95, hi95, color=color, alpha=0.12, lw=0)
    ax.fill_between(Z, lo68, hi68, color=color, alpha=0.30, lw=0)
    ax.plot(Z, med, color=color, lw=1.7, label=lab, zorder=3)

ax.errorbar(ACT_Z, ACT_S8X, yerr=ACT_ERR, fmt="o", ms=5.5, color="k", mfc="w", mew=1.4,
            capsize=3, zorder=6, label=r"ACT$\times$DESI LRG (Sailer+24; not in fit)")

ylo, yhi = ax.get_ylim()
for z_l, lbl in zip(HSC_Z, HSC_LBL):
    ax.axvline(z_l, color="gray", ls=":", lw=0.8, zorder=0)
    ax.text(z_l, ylo + 0.02 * (yhi - ylo), f" {lbl}", rotation=90, va="bottom", ha="left",
            fontsize=7, color="gray")
ax.set_xlabel(r"$z$", fontsize=12)
ax.set_ylabel(r"$S_8^{\times}(z)\equiv\sigma_8^{\Lambda{\rm CDM}}(\Omega_{\rm m}/0.3)^{0.4}\sqrt{\alpha(z)}$",
              fontsize=11)
ax.set_xlim(Z[0], Z[-1]); ax.set_ylim(ylo, yhi)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.grid(alpha=0.15, lw=0.5)
fig.tight_layout()
out = f"{OUTDIR}/S8x_of_z.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# ---------------------------------------------------------------- 正確性檢查
print("\n--- 正確性檢查 ---")
for lab in S:
    s, meta, cur = S[lab], M[lab], CUR[lab]
    print(f"\n[{describe(meta)}]")
    om, w = col(s, "om_m"), s.weights

    # 1) LCDM chain：band 必須水平以外的形狀只來自 (Om/0.3)^0.4（R ≡ 1）
    if meta["p"] is None:
        R_spread = np.ptp(cur / cur[:, :1], axis=1).max()
        print(f"  [1] LCDM：R(z) 最大偏離 = {R_spread:.2e}   (應為 0)")

    # 2) z=0：sigma8_lcdm * sqrt(alpha(0)) 必須等於 chain 的 sigma8（HSC 輸出的 DETG 值）
    s8 = col(s, "sigma8")
    beta = col(s, "beta_detg") if meta["has_beta"] else np.zeros(len(w))
    rec = col(s, "sigma8_lcdm") * np.sqrt(alpha(0.0, om, beta, meta["p"]))
    print(f"  [2] max|sigma8_lcdm*sqrt(alpha0) - sigma8| = {np.max(np.abs(rec - s8)):.2e}   (應 ~1e-15)")

    # 3) 與 chain 中 HSC 輸出的 S8_z_L1..3 比對（S8_z_L = sigma8_lcdm sqrt(alpha) D_LCDM(z) sqrt(Om/0.3)）
    j = np.argsort(w)[-200:]                      # 取權重最大的 200 個 sample 即可
    for i, zl in enumerate(HSC_Z):
        name = f"S8_z_L{i+1}"
        chain_val = col(s, name)
        if chain_val is None:
            continue
        mine = (col(s, "sigma8_lcdm")[j] * np.sqrt(alpha(zl, om[j], beta[j], meta["p"]))
                * D_lcdm(zl, om[j]) * np.sqrt(om[j] / 0.3))
        rel = np.max(np.abs(mine / chain_val[j] - 1))
        print(f"  [3] {name}: 重算 vs chain 最大相對差 = {rel:.2e}   (應 < 1e-2；D(z) 用 LCDM 積分式近似)")

    # 4) z=0 的值與外部 ACT×DESI 點
    print(f"  [4] S8x(z=0) = {np.median(cur[:, 0]):.4f} ± {np.std(cur[:, 0]):.4f}")
    for zz, v, e in zip(ACT_Z, ACT_S8X, ACT_ERR):
        b = np.interp(zz, Z, np.median(cur, axis=0))
        print(f"      z={zz:.2f}: model {b:.4f}, ACT×DESI {v:.4f} ({(v - b) / e:+.2f}σ_data)")

print(f"\nΩm(mean, {first}) = {om_ref:.4f}；Planck 換算因子 = {pl_conv:.4f}")
