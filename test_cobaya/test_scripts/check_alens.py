import sys
import glob
import numpy as np
from detg_common import load, stat, corr, has, root_path, describe, convergence, BURN

ROOT = sys.argv[1] if len(sys.argv) > 1 else "detg_hsc_act/run"
s, meta = load(ROOT)
print(describe(meta))
print(f"growth 設定（各 likelihood）：{meta['growth_opts']}\n")

for n in ["beta_detg", "Alens", "sigma8", "sigma8_lcdm", "S8", "S8_lcdm", "om_m", "H0", "logA", "tau", "A_planck"]:
    st = stat(s, n)
    if st:
        print(f"{n:14s} = {st[0]:.4f} +- {st[1]:.4f}")

if has(s, "Alens"):
    m, sd = stat(s, "Alens")
    print(f"\nA_L - 1 = {(m - 1) / sd:+.2f} sigma；r(beta, A_L) = {corr(s, 'beta_detg', 'Alens'):+.3f}")
    print("  |r| > 0.5：beta 與 A_L 在競爭同一個特徵（Planck lensed Cl 的平滑效應），需與 A_L=1 的 chain 對照")

print("\n各 chain 檔（burn-in 前的列數）：")
for f in sorted(glob.glob(root_path(ROOT) + ".*.txt")):
    if f.endswith((".progress.txt",)):
        continue
    with open(f) as fh:
        n = sum(1 for line in fh if not line.startswith("#"))
    print(f"  {f.split('/')[-1]:30s} n = {n}  (去掉前 {BURN:.0%} 後 {int(n * (1 - BURN))})")

cv = convergence(ROOT)
if "progress" in cv:
    print(f"\ncobaya .progress 最後一列：{cv['progress']}   （Rminus1 < 0.02 為收斂）")
for k, v in cv.items():
    if k != "progress":
        print(f"單變數 R-1 [{k}] = {v:.4f}")
if len(cv) == 0:
    print("\n(無法計算 R-1：需要 >= 2 條 chain 檔或 .progress)")



import sys
sys.path.insert(0, "/home/andromeda/cosmo_practice/test_cobaya/theory")  # growth_model.py 所在位置
from growth_model import GrowthModel
gr = GrowthModel("detg", 0.305, par=0.443, p=5)
print(gr.alpha(1.0))        # 預期 0.557 (= 1-β)
print(gr.ratio(0.0))        # 預期 0.746 (= √(1-β))
print(gr.f(0.3, 0.676))     # 預期 ≈ 0.30（ΛCDM f(0.3)=0.676）
print(gr.f(0.07, 0.559))    # 預期 ≈ -0.67