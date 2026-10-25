"""多條 DETG chain 的比較：不同 p、A_L 自由 vs 固定、DETG vs LCDM。

ROOTS 中找不到的 chain 會自動略過；各組圖只畫存在的 chain。
"""
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots
from detg_common import load, exists, stat, has, describe, OUTDIR

# ---------------------------------------------------------------- 設定（依實際 output 路徑修改）
ROOTS = {
    "p1_AL":   ("detg_p1_AL/run",    r"$p=1$, $A_L$ free"),
    "p3_AL":   ("detg_hsc_act/run",  r"$p=3$, $A_L$ free"),
    "p5_AL":   ("detg_p5_AL/run",    r"$p=5$, $A_L$ free"),
    "p1_AL1":  ("detg_p1_AL1/run",   r"$p=1$, $A_L=1$"),
    "p3_AL1":  ("detg_p3_AL1/run",   r"$p=3$, $A_L=1$"),
    "lcdm_AL": ("lcdm_hsc_act/run",  r"$\Lambda$CDM, $A_L$ free"),
}

# 每組圖：(檔名, 要比較的 key, 參數, markers)
FIGS = [
    ("tri_p_dependence",  ["p1_AL", "p3_AL", "p5_AL"],
     ["beta_detg", "sigma8_lcdm", "S8", "om_m", "f_ratio_z005"], {"beta_detg": 0.0, "f_ratio_z005": 0.0}),
    ("tri_Alens_p1",      ["p1_AL", "p1_AL1"],
     ["beta_detg", "S8", "logA", "Alens"], {"beta_detg": 0.0, "Alens": 1.0}),
    ("tri_Alens_p3",      ["p3_AL", "p3_AL1"],
     ["beta_detg", "S8", "logA", "Alens"], {"beta_detg": 0.0, "Alens": 1.0}),
    ("tri_detg_vs_lcdm",  ["p1_AL", "p3_AL", "lcdm_AL"],
     ["S8", "S8_lcdm", "om_m", "H0", "logA", "Alens"], {"Alens": 1.0}),
]

PARS = ["beta_detg", "sigma8", "sigma8_lcdm", "S8", "S8_lcdm", "S8_z_L2", "om_m", "H0",
        "logA", "tau", "Alens", "A_planck", "b1", "b2", "b3", "AIA", "dm_0", "dpz_0", "f_ratio_z005"]

# ---------------------------------------------------------------- 讀 chain
S, M = {}, {}
for k, (root, lab) in ROOTS.items():
    if not exists(root):
        print(f"略過（找不到）：{k} -> {root}")
        continue
    S[k], M[k] = load(root, label=lab)
    print("讀入", describe(M[k]))

# 檢查：同一組 A_L 對照的 p 必須相同
for a, b in [("p1_AL", "p1_AL1"), ("p3_AL", "p3_AL1")]:
    if a in M and b in M and M[a]["p"] != M[b]["p"]:
        print(f"!! {a} 與 {b} 的 p 不同（{M[a]['p']} vs {M[b]['p']}）：A_L 對照無效")

# ---------------------------------------------------------------- 圖
for fname, keys, pars, markers in FIGS:
    ks = [k for k in keys if k in S]
    if len(ks) < 2:
        print(f"略過 {fname}：可用的 chain 少於 2 條")
        continue
    common = [q for q in pars if all(has(S[k], q) for k in ks)]
    if len(common) < 2:
        print(f"略過 {fname}：共同參數不足")
        continue
    gp = plots.get_subplot_plotter(width_inch=8)
    gp.triangle_plot([S[k] for k in ks], common, filled=True,
                     legend_labels=[M[k]["label"] for k in ks],
                     markers={q: v for q, v in markers.items() if q in common})
    plt.savefig(f"{OUTDIR}/{fname}.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("saved", f"{OUTDIR}/{fname}.png")

# ---------------------------------------------------------------- 表
keys = list(S)
hdr = f"{'param':13s} " + " ".join(f"{k:>20s}" for k in keys)
print("\n" + hdr)
print("-" * len(hdr))
for n in PARS:
    row = f"{n:13s} "
    for k in keys:
        st = stat(S[k], n)
        row += f"{st[0]:9.4f} ±{st[1]:8.4f} " if st else f"{'(fixed/n.a.)':>20s} "
    print(row)

# A_L 對照的結論
print("\nA_L 自由 vs 固定：beta 的移動量（以 A_L 自由的 sd 為單位）")
for a, b in [("p1_AL", "p1_AL1"), ("p3_AL", "p3_AL1")]:
    if a in S and b in S:
        (m1, s1), (m2, _) = stat(S[a], "beta_detg"), stat(S[b], "beta_detg")
        print(f"  {a} vs {b}: Δbeta = {m1 - m2:+.4f} = {(m1 - m2) / s1:+.2f} σ"
              + ("   → Planck lensed C_l 的近似影響結論，以 A_L 自由為準" if abs(m1 - m2) / s1 > 0.3 else "   → 可忽略"))
