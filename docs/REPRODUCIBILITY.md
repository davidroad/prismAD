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

Release 1.2.1 synchronizes reporting documentation and package metadata.
It does not change scientific algorithms or repeat model training. The Local
reporting helpers were checked against the final supplementary tables.
Current Public and Local Code directories must be byte-identical to the same
published commit. The complete author package records that commit and the
private input identities without publishing the inputs.


## Release 1.3.0 diagnostic extension
The two added commands audit existing entry/outcome records and fitted-term
comparisons. They do not modify the prediction model or reproduce the entire
raw-data pipeline by themselves. The Local runner includes both steps in a
fresh complete run. Synthetic tests validate defined behaviors; private
execution against study artifacts is recorded separately. Earlier development
had already inspected evaluation folds, so fixed refitting does not make them
an untouched test set.
