# Statistical and biological interpretation
The unit for primary evaluation is one first eligible visit per participant.
Repeated eligible visits contribute during training. Participant bootstrap samples
are drawn within fold and retain paired predictions and fixed censoring weights.
Nominal percentile intervals use 500 draws. Exploratory two-sided P values use
9,999 null-centered paired bootstrap draws (seed 42); these intervals are not
inversions of the tests and remain nominal.

The primary AP comparison uses current measurements. First-MCI forecasts and
AD-dementia-after-MCI forecasts are distinct targets. AP depends on event frequency.
Three- and five-year models are independently fitted, not a joint cumulative-risk curve.
IPCW depends on the specified study/age censoring assumptions; death and non-AD
dementia were not separately modeled.

## Multiple-comparison families
- 28 first-eligible-visit AP contrasts; 16 all-eligible-visit history comparisons separately.
- Eight PRISM-AD PRS increments: two tasks, two horizons, AP and AUROC.
- Four HC3 omnibus comparisons of C across shared PRS quintiles.
- 128 targeted contribution contrasts; unavailable tests keep P=1 correction slots.
- Twelve shared-C AD–CN contrasts: pooled, ten deciles and expanded sensitivity.
- Four education comparisons, including unavailable comparisons.

Ordinary *, ** and *** indicate BH q<0.05, 0.01 and 0.001.
(*) indicates nominal P<0.05 with BH q>=0.05.
No PRS-increment or PRS-quintile omnibus test reached q<0.05.

## Core shared-C comparison
Shared cognitive-risk deciles are defined in all 893 initially CN participants.
The display contains 701 last-CN, 97 last-MCI and 95 later-AD participants.
AD–CN inference uses 796 participants and adjusts signed G for C-decile indicators,
continuous C and prediction fold. Expanded adjustment adds age, age squared,
sex, study and PC1–5. Inference uses HC3 covariance and residual-df t intervals.
Deciles with fewer than five AD or five CN participants are not estimable.
Exponentiating the AD coefficient gives a modeled genetic-factor ratio.
The pooled estimate is 1.134 (nominal 95% CI 1.072–1.200, BH q=0.0000734).
It is not an odds ratio for developing AD. Binning on C explains the increase
in C across deciles. It does not imply biological genetic–cognitive compensation.

## Supporting context and optimization
Education is a reserve proxy. Its phenotype comparison and conditional model term
address different questions. PET documents amyloid during normal cognition;
the median PET-to-last-CN interval was 0.22 years, limiting sustained-resilience inference.
Training curves cover 24 original fits and 1,440 epochs. Training BCE on the fixed
IPCW resample and normalized IPCW validation BCE use different visit/weight constructions.
Loss diagnostics use all eligible known-outcome validation visits, whereas primary
performance uses first eligible visits. Late validation increases permit possible
overfitting. No validation-based checkpoint selection was performed.
