# Corrected execution graph
Configure authorized inputs outside the checkout in a private JSON copied from
config/input.example.json. Receiving code does not grant data access. Use a new
external work directory, never a public checkout or distribution folder.

## Input score version
Baseline and genotype-roster inputs must contain the same corrected PRS.
The upstream R scripts estimate SE-derived effective information and match the
LDSC denominator to the LD variants. Use a new PRS_OUTPUT_DIR. Existing
unregistered LDSC/auto caches are rejected and historical files are preserved.
The reconstruction starts from post-QC genotypes, not raw VCFs.

## Full downstream workflow
The Local collaborator package provides scripts/reproduce_corrected.py as the
complete orchestration entry point. For manual execution, keep this exact graph
and point every configuration role to the corresponding fresh output:

1. features -> work/prediction; groups -> work/groups.
2. eligibility-audit -> work/eligibility_audit, then train --preflight -> work/prediction.
3. train both clinical tasks -> work/prediction.
4. Create work/no_prs/LOCAL_RESTRICTED and copy the successful same-environment
   PREFLIGHT.json to work/no_prs, or run its preflight.
5. train --without-prs --features work/prediction for both tasks -> work/no_prs.
6. metrics --bootstrap -> work/prediction.
7. paired-inference -> work/paired; genetic-comparison -> work/genetic.
8. interpret -> work/interpretation; education -> work/education; pet -> work/pet.
9. profile-replay -> work/profiles, including the full first-eligible CN ledger.
10. export -> work/SourceData; profile-export -> work/SourceData/tables.
11. current-statistics -> work/statistics; cognitive-strata -> work/cognitive_strata.
12. model-scale-checks -> work/model_scale; followup-sensitivity -> work/followup; panel-diagnostics -> work/panel_diagnostics; then figure-inputs -> work/SourceData.
13. learning-curve-replay -> work/optimization. Copy its learning_curves.csv and
    passing replay_verification.json into work/SourceData/optimization.
14. render --figure all --source-data work/SourceData --output work/Figures.

All analysis commands except render receive --config and --output. Select
CN_MCI or MCI_AD with train --task. Outcomes and clinical inputs are constructed
before any horizon-specific model is fitted. The full graph includes the
matched clinical-only controls. Do not mix corrected PRS with legacy forecasts.

## GPU and diagnostic settings
Allocate one full training worker per GPU. CUDA_VISIBLE_DEVICES selects physical
devices, while a worker uses its logical cuda:0. Retain the recorded environment,
seeds, preprocessing and 60 fixed epochs. The three participant-validation folds
are excluded from their respective training rounds. Earlier development fold 0
is excluded from these analyses.

Loss replay can be split with PRISM_AD_DIAGNOSTIC_TASK. Keep its output in
work/optimization/CN_MCI and work/optimization/MCI_AD. Concatenate both loss CSVs
and combine the two passing run_audits lists before rendering the final loss
figure. Replay stops unless losses, final parameters and probabilities match.
The monitoring does not select a different checkpoint.

## Protected outputs
The run creates participant records, individual scores/contributions, model
weights, empirical summaries and logs. All remain private. Public and Local
distributions contain none of these outputs. Private plotting JSON records, if
generated, are also controlled derivatives and cannot be bundled as code.


## Reconcile final publication reporting
The separate private Local package supplies two additional table exporters.
After a complete run, build_publication_tables.py consumes the fresh work root,
while build_reporting_tables.py consumes its resolved private configuration.
Use the commands in Local/README.md and keep outputs outside both code packages.
Copy only publication-format TeX into a private article build; do not commit
generated tables, receipts or underlying summaries.

For a code/documentation synchronization, run the public release guard and
synthetic checks, then the Local file verifier and input check. This verifies
the synchronized release without claiming a new raw-data or model rerun.

## Eligibility and model-scale diagnostics (v1.3.0)

    python -m prism_ad eligibility-audit --config /PRIVATE_CONFIG.json --output /PRIVATE_NEW_AUDIT
    python -m prism_ad model-scale-checks --config /PRIVATE_CONFIG.json --output /PRIVATE_NEW_SCALE

Run eligibility-audit after features and groups have created metadata and course
indices. It checks all saved risk visits against the full original history,
reconstructs target/competing-event outcomes and reports aggregate accounting.
An AD record after prediction followed by CN or MCI does not itself invalidate
entry. The retrospective profile sample has additional linkage and course rules.

Run model-scale-checks after profile-replay and cognitive-strata. It consumes
current_profile_components, profile_join_keys and genotype_roster. Within-fold
G contrasts use continuous C and the common C deciles. The pooled raw-PRS
contrast also adjusts for fold. The reference is the complete configured
genotype roster mean and population SD (ddof=0). HC3 residual-df t inference and
a separate four-test BH family leave the original twelve-test family unchanged.
These post hoc checks reuse fitted C, participants and overlapping development
samples. They neither align every fitted contribution nor provide independent
genetic evidence, fixed-horizon AD incidence estimates or external validation.

All diagnostic outputs, including aggregate CSV/JSON, are controlled derivatives.
Keep them outside code packages and Git repositories. The Local table exporter
can append the model-scale results to publication Table 4 in a private article
build. No generated empirical results belong in this repository.

## Follow-up endpoint and frozen plan
Configure first_CN_components from the restricted full-first-CN parquet written
by profile-replay. Configure followup_analysis_plan and
followup_plan_amendment to authorized private method records. These are not
bundled cohort artifacts. The Local file index names their server locations.
The amendment binds the plan hash and the inclusive-horizon left-limit
censoring convention. Supply execution.survival_Rscript with R survival installed.

    python -m prism_ad followup-sensitivity --config /PRIVATE_CONFIG.json --output /PRIVATE_NEW_FOLLOWUP

The command uses all eligible first-CN entries, without future clinical-course
selection. First AD is terminal even if a later assessment reverts, while MCI
is intermediate. A first undated AD stops follow-up at the preceding dated
non-AD assessment. A new AD-specific reverse-Kaplan--Meier censoring model is
used for the five-year comparison. Original first-MCI weights are not reused.

The Cox analysis uses R survival Efron ties, its default near-tie correction and
robust sandwich inference. Report the rank-time PH diagnostics together with
the working-model estimate. A nonsignificant term-specific diagnostic does not
establish proportional hazards. The five-year WLS effect is a raw-score SD
difference, not clinical odds. Keep the two-test BH family separate from the
original families, and retain unavailable outcomes as unknown rather than controls.
Do not select an alternative model to obtain a favorable result.

All ledgers, CSVs, diagnostics, regression artifacts, source hashes and private
plans remain outside Public/Local and all Git repositories. Reconstructing this
analysis does not establish external validation, competing-risk AD incidence,
biological onset or diagnostic lead time.

## Post hoc panel diagnostics (v1.5.0)

    python -m prism_ad panel-diagnostics --config /PRIVATE_CONFIG.json --output /PRIVATE_NEW_PANEL_DIAGNOSTICS

Run after first-visit predictions and current_profile_components exist. Supply
panel_analysis_plan as a separately authorized frozen revision-method record.
It defines four calibration-slope and six marginal clinical-course test slots.
The plan was fixed for this revision, after the underlying study and earlier
descriptive figures existed; it is not prospective preregistration.

The command estimates intercept and slope against individual saved logit
forecasts with positive IPCW known outcomes. Its empirical participant-score
sandwich uses HC1 n/(n−2), normal intervals and two-sided slope = 1 tests.
Intervals condition on the saved forecasts and censoring weights; they do not
propagate prediction-model fitting or weight-estimation uncertainty. Diagnostic
fits are not applied to recalibrate the forecasts. A nonsignificant slope test
does not establish calibration across the full risk range.

Marginal C/G comparisons use fold-specific midranks, tie-corrected variances,
inverse-(pair-specific fold size + 1) weights and two-sided van Elteren normal
tests. The signed rank-biserial summary is twice weighted pairwise concordance
minus one; positive values indicate larger values in the second named group.
These distributional tests are not pure median tests. They do not adjust for
C, age, study or follow-up and do not estimate causal or clinical AD risks.
They differ from earlier within-course PRS-quintile tests and the analysis
conditioning on comparable cognitive contributions.

All ten slots share one new BH family. Nonestimable slots remain in the family
with P = 1. Stars denote BH q thresholds; (*) denotes nominal-only P < 0.05.
The plotting code consumes the resulting labels without selecting tests.
All individual points, diagnostic arrays, summaries and test tables remain
private outside code packages. Set panel_diagnostics_directory to the executed
output before figure-inputs or publication table export. A fresh Local run
redirects it into its new work root.
