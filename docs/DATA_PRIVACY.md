# Distribution boundary
Keep all ADSP/PHC portal files and participant-level derivatives in private,
authorized storage. Never commit raw genotypes, dosages, cognitive or clinical
records, participant identifiers, rosters/linkage keys, individual PRS, per-person
model outputs, trained weights, anonymized individual examples or reconstruction
objects. Analysis outputs are private by default.

Only code, environments, empty input templates, synthetic tests and the manually
reviewed model schematic are included in this release. Source accessions and
field names describe interfaces; they do not redistribute portal data.

Run scripts/check_public_release.py before distributing the directory and
scripts/check_public_release.py --staged before committing. The checker rejects
data-container formats, tabular datasets, unapproved JSON/image files, known
identifier/credential patterns and private paths. It does not determine source-
cohort data-use permissions, and a new public artifact still requires review.

Public figures or summary statistics, if separately authorized, need explicit
artifact review and a documented distribution boundary. They are not included
by copying the private analysis directory or an earlier complete bundle.
