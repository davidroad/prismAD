# Code and workflow index
All paths below are relative to the Code/repository root. No controlled inputs
or individual derivatives are included. The separate local collaborator bundle
adds an author-side server-path index and configuration references.

| Purpose | Code |
| --- | --- |
| Command entry and local configuration | src/prism_ad/cli.py; runtime.py; input_io.py |
| Clinical transition definitions and observation | src/prism_ad/transitions.py; build_features.py |
| Training-only transformations and past assessments | src/prism_ad/features.py; constants.py |
| Nonlinear clinical functions and interactions | src/prism_ad/native.py; threshold.py |
| Complete PRISM-AD model and history component | src/prism_ad/model.py |
| Fitting and licensed comparator interface | src/prism_ad/training.py; inference.py |
| Matched clinical-only comparison | src/prism_ad/genetic_control.py; genetic_metrics.py |
| Prognostic metrics and paired comparisons | src/prism_ad/metrics.py; paired_inference.py |
| Clinical-course groups | src/prism_ad/groups.py; phenotypes.py; cognitive_course.py |
| Cognitive/genetic contribution accounting | src/prism_ad/contributions.py; interpret.py |
| Frozen-checkpoint profile replay | src/prism_ad/profile_replay.py |
| Shared-cognition-strata inference | src/prism_ad/cognitive_strata.py |
| Diagnostic-history and endpoint audit | src/prism_ad/eligibility_audit.py |
| Descriptive publication summaries | src/prism_ad/publication_summaries.py |
| First-AD follow-up sensitivity | src/prism_ad/followup_sensitivity.py |
| Post hoc slope/rank diagnostics | src/prism_ad/panel_diagnostics.py; marginal_tests.py; publication_panel_diagnostics.py |\n| Private panel table reporting | src/prism_ad/panel_reporting.py |\n| New calibration/marginal panels | src/prism_ad/figures_current/calibration_panels.py; marginal_panels.py |
| Within-fold and raw-PRS sensitivity | src/prism_ad/model_scale_checks.py |
| Registered PRS and quintile tests | src/prism_ad/current_statistics.py |
| Summary export for private analysis | src/prism_ad/profile_export.py; export_metrics.py |
| Education and amyloid PET | src/prism_ad/education.py; biomarkers.py; pet.py |
| Retrospective optimization replay | src/prism_ad/learning_curve_replay.py |
| Figure construction with private inputs | src/prism_ad/render.py; figures_current/; plot_records.py |
| Optional PRS construction | upstream/step_zero_all_samples.R; upstream/step1a_all_samples.R |
| Synthetic scientific and release checks | tests/; src/prism_ad/demo.py; scripts/check_public_release.py |
| Environment specifications | environments/ |
| Empty input interface | config/input.example.json |

First run the synthetic example and tests. Real-data analysis requires authorized
input paths and the original allocations/checkpoints. Use features and groups
before new model fitting. Metrics, contribution analyses and private figures can
also use the original frozen server artifacts; these inputs are never copied
into the public code release.

## Corrected figure-input assembly
prepare_figure_inputs.py stages current private figure inputs from the fixed execution graph.
figures_current/figure5_native.py rebuilds all Figure 5 panels from those fresh results.
figures_current/figure5_layout.py and panel_layout.py contain data-free layout geometry only.
No real plotting records or empirical tables are distributed.

## Private Local reporting add-ons
The full collaborator bundle, separately from this public repository, provides
scripts/build_publication_tables.py for publication Tables 1--10 and
scripts/build_reporting_tables.py for characteristics, follow-up and calibration
tables. Its CODE_FILE_INDEX.json binds their exact private server sources.
The public code exposes the underlying analyses; generated publication tables,
server paths, input identities and receipts remain outside GitHub.
