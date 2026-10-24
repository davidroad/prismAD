# Reproducibility scope
This release contains code, environments, schemas, synthetic checks and a data-free
Figure 1a schematic. It contains no controlled cohort data or derivatives.
The corrected workflow estimates GWAS effective information from supplied SEs,
uses the matched LD variant universe for LDSC normalization and guards against
reusing incompatible LDSC/auto caches. The reconstruction used existing post-QC
genotype/LD objects. Raw-VCF QC was not repeated and discovery overlap remains unknown.

The complete downstream model grid, clinical-only controls, contribution tests,
biological-context analyses and all figures were regenerated privately.
Optimization diagnostics require exact replay of each saved loss, parameter and
held-out prediction. Validation is internal and clinical benefit is untested.

Public files exclude real observations, empirical plotting inputs, participant
summaries, fitted weights, individual examples and private paths.
Local adds path/hash metadata and orchestration, not datasets or data access.
