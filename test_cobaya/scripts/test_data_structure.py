import numpy as np
from astropy.io import fits

FITS = '/home/weichen/cosmo_practice/test_cobaya/fiducial/dataset_hsc_y3.fits'
with fits.open(FITS) as hd:
    hd.info()
    print()
    print(hd['nz_lens'].columns)
    print(hd['nz_lens'].data['Z_MID'][:5])
    print(repr(hd[0].header))