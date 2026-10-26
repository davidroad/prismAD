# PRISM-AD

Release 1.2.1 (26 October 2026) aligns documentation with the final manuscript
and supplementary reporting. The scientific implementations and fitting settings
are unchanged from the corrected-PRS reconstruction.

Code for interpretable cognitive prognosis with explicit clinical and genetic
contributions. PRISM-AD learns nonlinear clinical functions and retains a
distinct PRS term. Its predicted probability is sigmoid(C + G + R), where C
contains cognitive main functions, G the genetic term and R all remaining terms.
The primary outcome is first recorded MCI after normal cognition. AD dementia
after MCI is a separate secondary outcome, with three- and five-year horizons.

![Figure 1a. PRISM-AD model architecture; a schematic without participant data.](assets/figure1a.png)

## Install and run synthetic checks
Use a separate Python 3.11 model environment. Full neural-model checks require
PyTorch, while the basic synthetic example uses NumPy and pandas.

    git clone https://github.com/davidroad/prismAD.git
    cd prismAD
    python -m pip install -e '.[analysis]'
    python -m prism_ad demo

For the recorded model environment and neural scientific tests:

    python -m pip install -r environments/model-requirements.txt
    python -m pip install --no-deps -e .
    python -m prism_ad demo --with-model
    python -m unittest discover -s tests -v

## Controlled-data boundary
ADSP/PHC source files and participant-level derivatives are not distributed here.
This includes genotype/variant data, roster and linkage files, individual PRS,
clinical or cognitive records, per-person predictions/contributions, model
checkpoints, anonymous individual examples and derived reconstruction objects.
The supplied tests and demo generate synthetic data.

Obtain any required controlled inputs separately through the applicable
authorized access process. Keep them and all analysis outputs in private local
storage. Input schemas and placeholder configuration paths describe the interface
without supplying the underlying records. See [INPUTS.md](docs/INPUTS.md),
[RUNNING.md](docs/RUNNING.md) and [DATA_PRIVACY.md](docs/DATA_PRIVACY.md).

This release contains code, configuration templates, environments and the
data-free Figure 1a schematic. It does not include cohort result tables, figure
source data, plotted individual records, manuscript files or full-precision
reconstruction inputs. Figure and statistical commands require separately
authorized local inputs. Code availability does not establish a fully executable
raw-data reproduction or grant access to the controlled datasets.

## Code navigation
See [CODE_INDEX.md](docs/CODE_INDEX.md) for the module-to-workflow mapping.

## Release check
Before distributing a directory or committing staged changes:

    python scripts/check_public_release.py
    python scripts/check_public_release.py --staged

See LICENSE.txt and THIRD_PARTY_NOTICES.md for reuse and third-party terms.

## Reproduction modes and current reporting
The public checkout provides the analysis implementation and synthetic checks.
The separate, private Local collaborator package adds checked server paths,
input-identity indexes and execution wrappers. It includes:

- scripts/reproduce.py for validating and replaying authorized frozen inputs.
- scripts/reproduce_corrected.py for complete refitting in a new private directory.
- scripts/build_publication_tables.py and scripts/build_reporting_tables.py for
  the final publication-format tables, including characteristics, follow-up and
  calibration reporting.

These Local paths are not files in this GitHub checkout. No Local configuration,
input hashes, result tables or manuscript documents are distributed on GitHub.
The final reporting helpers were executed and reconciled with the manuscript
separately from the public synthetic checks. A documentation update does not
require rerunning the unchanged model grid. Use the complete private refit only
when intentionally reproducing or changing the authorized inputs or analysis.
