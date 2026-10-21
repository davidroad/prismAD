# Input contracts
No ADSP/PHC source files or participant-level derivatives accompany this release.
The cognition and genotype resources require separately authorized access.
The synthetic demo and tests do not use cohort data.

config/input.example.json supplies placeholder paths only. Its checksum mapping
is empty; authorized users can provide their own local expected hashes.
- folds: original participant allocation; private.
- baseline, cognition, genotype_roster and amyloid_pet: controlled input interfaces.
- tabpfn_checkpoint: separately obtained licensed model.
- prediction_directory and no_prs_directory: private fitted artifacts.
- cn_course_indices and cn_course_groups: private follow-up/group reconstruction.
- prior_targeted_contributions and current_profile_components: private model outputs.
- profile_join_keys: private subject/visit linkage.
- interpretation, education, paired, genetic and PET directories: private outputs.
- source_data_directory: private local figure inputs, not a supplied dataset.

Actual input files, allocations, identifiers, per-person PRS or contributions,
and model files must not be added to GitHub or the distributable archive.
Optional plotting commands require local authorized analysis outputs.
A public GWAS accession identifies an upstream publication and does not provide
access to the controlled target-cohort genotypes.
