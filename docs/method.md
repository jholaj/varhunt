# Method

How varhunt turns TESS light curves into variable stars that no catalogue lists, and how far to trust the result.

## How a star becomes a candidate

1. Light curves come from the MIT Quick Look Pipeline. The raw aperture flux is used, the first day after every gap is dropped and slow drifts are removed with a running median. Signals near the window of the median are protected by detrending again with the signal model divided out.
2. Sinusoids are found by Lomb-Scargle with prewhitening and need an amplitude S/N of 5 against the local noise. Eclipses are found by a box search up to half the baseline and kept only when they explain the light curve better than the sinusoids. When every other eclipse is shallower, the period is doubled and the shallow ones become the secondary eclipse.
3. The strongest signal has to look alike in both spacecraft orbits and has to lie inside the searched periods, a star whose strongest signal is slower is marked slow.
4. A frequency shared across the whole field is instrumental. A frequency shared by neighbours belongs to the star with the most variable flux, the others are just its echoes.
5. No VSX or Gaia DR3 variable may sit within one TESS pixel.
6. The pixel check fits the signal to every pixel of a TESScut cutout, removes the sky's share and requires the rest to sit on the target and not to be image motion. The signal is located by a Gaussian fitted to the pixel map. Its noise comes from the same signal shape stretched to nearby frequencies, so pointing jitter and scattered light are measured, not assumed.

## When a candidate is sure

- Its S/N is at least 10 and another TESS sector repeats it, or at least 7 and two more sectors repeat it.
- The signal sits within 0.3 pixel of the target with a centroid error below 0.15 pixel, it sits on the target in another sector and nowhere else.
- It stays when the detrending is three times wider, so it is not what a slower variation left behind.
- No variability catalogue in VizieR or SIMBAD lists the star within 5 arcsec, entries of other stars within 30 arcsec are listed for a human to judge.
- No unresolved Gaia source at the signal position could have faked it.

Confirming also refines the period over all sectors, anchored on the longest unbroken run so the cycle count across gaps is safe or flagged.

## Preparing a VSX submission

`varhunt vsx` gathers what the New Star Form asks for. The position is the Gaia DR3 one at J2000, the maximum is a V magnitude from APASS DR9 or from Gaia G and BP-RP, the minimum is the TESS amplitude. Eclipsing binaries get a linear ephemeris fitted to every eclipse in the TESS-SPOC and QLP light curves, other variables the period from all sectors. The TESS-SPOC PDCSAP light curves, which VSX moderators know best, are checked for the same signal and used for the plots. Epochs are converted from BJD(TDB) to the HJD the form expects.

VSX accepts one submission from a new user until it is approved, then five a week. For many discoveries it recommends a journal paper or a bulk import.

## How well it works

`varhunt benchmark` with the default seed, 43 random VSX stars of 9 to 12.5 mag that have QLP data:

| VSX type | Stars | Period within 1 % | Also half or double | Type hint right |
| --- | ---: | ---: | ---: | ---: |
| EW | 9 | 7 | 8 | 8 |
| DSCT | 12 | 8 | 10 | 9 |
| RRAB | 10 | 10 | 10 | 10 |
| EA | 12 | 10 | 11 | 9 |

One EA counted as wrong is ASASSN-V J033545.95-591922.1, where TESS shows an eclipse only every second VSX cycle, so its VSX period is half the true one.

Injected sinusoids in a sector 72 field (`varhunt inject`, 500 trials) are always recovered from 1 ppt for stars brighter than Tmag 12. Fainter stars need about 3 ppt and reach 86 to 100 %, the misses sit in a few light curves ruined by outliers.

## Limits

QLP covers stars down to about Tmag 13.5. Periods are searched up to `--max-period` (2 days by default), raising it also widens the detrending. A TESS pixel is 21 arcsec, so a passed pixel check only rules out resolved neighbours, Gaia sources closer than half a pixel are listed in the summary. Type hints are rules of thumb, the TIC temperature behind them is too low for reddened stars.
