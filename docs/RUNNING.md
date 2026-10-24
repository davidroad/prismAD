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
2. train --preflight -> work/prediction.
3. train both clinical tasks -> work/prediction.
4. Create work/no_prs/LOCAL_RESTRICTED and copy the successful same-environment
   PREFLIGHT.json to work/no_prs, or run its preflight.
5. train --without-prs --features work/prediction for both tasks -> work/no_prs.
6. metrics --bootstrap -> work/prediction.
7. paired-inference -> work/paired; genetic-comparison -> work/genetic.
8. interpret -> work/interpretation; education -> work/education; pet -> work/pet.
9. profile-replay -> work/profiles.
10. export -> work/SourceData; profile-export -> work/SourceData/tables.
11. current-statistics -> work/statistics; cognitive-strata -> work/cognitive_strata.
12. figure-inputs -> work/SourceData.
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


## Complete corrected workflow
Authorized collaborators can use the sibling Local package's
scripts/reproduce_corrected.py to run the complete graph in a new external private
directory. Baseline and genotype-roster inputs must contain the corrected PRS.
Do not point these inputs to historical prediction directories.

After export and profile-export, run current-statistics and cognitive-strata,
then figure-inputs. The fixed graph uses work/prediction, work/no_prs,
work/interpretation, work/education, work/paired, work/genetic, work/pet,
work/profiles, work/statistics, work/cognitive_strata and work/SourceData.
Point every configuration role to the new outputs. figure-inputs assembles the
private plotting summaries from this graph.

The no-PRS output needs its own PREFLIGHT.json and LOCAL_RESTRICTED folder.
Run its preflight or copy a successful identical-environment preflight receipt.

Loss replay can be split by setting PRISM_AD_DIAGNOSTIC_TASK to one task and
CUDA_VISIBLE_DEVICES to its allocated GPU. Combine both passing audit lists and
loss tables before rendering Supplementary Figure 3. This affects orchestration,
not the model, optimizer, epoch selection or scientific settings.
