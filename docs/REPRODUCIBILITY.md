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


## Release 1.4.0 figure and follow-up extension
The model core is unchanged. The full-first-CN export precedes retrospective
profile exclusions. New panel summaries are descriptive and preserve native
fold scales. Follow-up sensitivity fits additional association regressions under
a private frozen plan, with full proportional-hazards/censoring diagnostics.
No assumption check is a guarantee that ascertainment or external validity has
been resolved. The public release includes code and synthetic tests, not the
private empirical estimates or full-first-CN ledger.

## Release 1.4.1 publication numbering

Current supplementary exports follow first citation in the manuscript. S1 is
optimization, S2 age and S3 PRS increments. Table display numbers are mapped
from stable scientific output identifiers in publication_numbering.py. This
reporting change preserves the fitted models, calculations and test families.

## Release 1.4.2 descriptive panel graphics

Calibration is drawn as separate three-/five-year bin markers without connecting
or smoothing. Marginal C/G distributions use fold-specific q25–q75 boxes and
median lines, with no whiskers, pooled fitted scales or new inference. Rendering
uses the same authorized summary inputs; no model fitting or recalibration is
introduced by this display change.

## Release 1.5.0 post hoc diagnostic and swarm extension

The panel-diagnostics command fits four calibration diagnostics and computes
six within-fold rank comparisons in one separate ten-test BH family. It uses
the original saved first-visit forecasts and current profile components.
Calibration diagnostics do not alter probabilities; their uncertainty is
conditional on saved model outputs and IPCW. Marginal clinical-course
comparisons retain fold-specific scales and do not adjust for age, study,
follow-up or cognitive contributions. They are exploratory, rather than tests
of added predictive value or external validation.

All individual contributions are displayed with deterministic horizontal swarm
packing and no changes to their signed values. Private table exporters consume
the same ten-test output used for figure annotations. These empirical inputs,
results and individual points remain outside the public/local code folders.
Earlier figures a–d, original prediction-model fits and inference families
remain intact. Run the synthetic suite, the public release guard and the
private Local input checks; do not infer cohort reconstruction from synthetic
checks alone.
