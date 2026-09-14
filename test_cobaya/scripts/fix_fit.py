import numpy as np
from astropy.io import fits

BASE = '/home/weichen/cosmo_practice/test_cobaya/fiducial'
FITS = f'{BASE}/dataset_hsc_y3.fits'
OUT  = f'{BASE}/dataset_hsc_y3_fixed.fits'
DD   = f'{BASE}/dataset'

# 讀三個 lens bin 的 n(z)
nzl = [np.loadtxt(f'{DD}/nzl_z{i}_100bin.dat') for i in range(3)]
for i, d in enumerate(nzl):
    print(f'z{i}: {d[:,0].min():.4f} – {d[:,0].max():.4f}')

# 共同網格(等寬,方便算 Z_LOW/Z_HIGH)
nb = 400
edges = np.linspace(0.10, 0.75, nb + 1)
z_low, z_high = edges[:-1], edges[1:]
z_mid = 0.5 * (z_low + z_high)

cols_nz = []
for d in nzl:
    n = np.interp(z_mid, d[:, 0], d[:, 1], left=0.0, right=0.0)
    n = n / np.trapezoid(n, z_mid)          # 正規化
    cols_nz.append(n)
    print(f'  mean z = {np.average(z_mid, weights=n):.4f}')

new = fits.BinTableHDU.from_columns([
    fits.Column(name='Z_LOW',  format='D', array=z_low),
    fits.Column(name='Z_MID',  format='D', array=z_mid),
    fits.Column(name='Z_HIGH', format='D', array=z_high),
    fits.Column(name='BIN1',   format='D', array=cols_nz[0]),
    fits.Column(name='BIN2',   format='D', array=cols_nz[1]),
    fits.Column(name='BIN3',   format='D', array=cols_nz[2]),
], name='nz_lens')

hd = fits.open(FITS)
idx = [i for i, h in enumerate(hd) if h.name == 'nz_lens'][0]
new.header['HISTORY'] = 'nz_lens rebuilt from nzl_z[0-2]_100bin.dat'
hd[idx] = new
hd.writeto(OUT, overwrite=True)
hd.close()

# 驗證
with fits.open(OUT) as h:
    d = h['nz_lens'].data
    for b in ['BIN1', 'BIN2', 'BIN3']:
        print(f'{b}: mean z = {np.average(d["Z_MID"], weights=d[b]):.4f}')