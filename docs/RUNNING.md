# Execution order
Commands below assume the code directory as the working directory. Configure authorized local
paths in config/input.local.json. Outputs containing participant information remain
private and are excluded from the distributed package.

## Original analysis workflow
    python -m prism_ad features --config config/input.local.json --output work/prediction
    python -m prism_ad groups --config config/input.local.json --output work/groups
    python -m prism_ad train --config config/input.local.json --output work/preflight --preflight
    python -m prism_ad train --config config/input.local.json --output work/prediction --task CN_MCI
    python -m prism_ad train --config config/input.local.json --output work/prediction --task MCI_AD
    python -m prism_ad train --config config/input.local.json --output work/without_prs --features work/prediction --task CN_MCI --without-prs
    python -m prism_ad train --config config/input.local.json --output work/without_prs --features work/prediction --task MCI_AD --without-prs
    python -m prism_ad metrics --config config/input.local.json --output work/prediction --bootstrap
    python -m prism_ad paired-inference --config config/input.local.json --output work/paired
    python -m prism_ad genetic-comparison --config config/input.local.json --output work/genetic
    python -m prism_ad interpret --config config/input.local.json --output work/interpretation
    python -m prism_ad education --config config/input.local.json --output work/education
    python -m prism_ad pet --config config/input.local.json --output work/pet

Do not run GPU tasks simultaneously unless separate devices were allocated.
The original grid contains 96 +PRS fits and 48 clinical-only fits. Three validation
folds are held out in turn; the remaining eight of folds 1–9 train each model.
Fold 0 was used in earlier development and is excluded here.
Seven variants share the same IPCW resample. Native-weighted HGB instead uses
all known-outcome training visits and their weights directly.

## Current contribution and inference extensions
    python -m prism_ad profile-replay --config config/input.local.json --output work/current_profiles
    python -m prism_ad profile-export --config config/input.local.json --output work/profile_aggregates
    python -m prism_ad current-statistics --config config/input.local.json --output work/current_statistics
    python -m prism_ad cognitive-strata --config config/input.local.json --output work/cognitive_strata

The profile replay reads original checkpoints and checks saved predictions/ledgers,
including the previous 688-person overlap. It writes a protected 893-person
component table and private join keys. Profile-export releases aggregate summaries
only. Current-statistics computes the eight PRS and four PRS-quintile tests.
Cognitive-strata computes the pooled, ten within-decile and expanded HC3 contrasts,
retaining all twelve correction slots. These stages do not fit prognostic models.
Point the configuration to the earlier stage outputs before running downstream steps.

## Retrospective optimization monitoring
    python -m prism_ad learning-curve-replay --config config/input.local.json --output work/loss_replay

This optional command uses the original data/order/resamples/seeds and checkpoints,
a compatible CUDA/Python/Torch environment and the original 60-epoch optimizer.
It stops unless every original online loss, final parameter and prediction matches.
It writes diagnostics to the chosen output directory and does not select epochs
or replace original checkpoint files. It was not rerun during this packaging task.

## Figures from authorized local analysis outputs
    python -m prism_ad render --figure all --source-data ../private_local/figure_inputs --output ../work/figures

Fresh output directories avoid overwriting a prior execution. This command needs
no additional model fitting when authorized aggregate inputs are staged. These inputs are not distributed here.
