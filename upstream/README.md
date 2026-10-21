# Optional upstream PRS construction

The two R files retain the existing all-sample LDpred2-auto implementation. They start from **QC-processed European-ancestry PLINK genotypes**, not raw sequencing reads.

Set `HM3_MAP_FILE`, `SUMMARY_STATS`, `FINAL_BFILE`, `GENETIC_MAP_DIR`, `PRS_OUTPUT_DIR` and optionally `PRS_NCORES` / `PRS_R_LIB`. Run `step_zero_all_samples.R` then `step1a_all_samples.R` with Rscript in the original PRS environment. `SUMMARY_STATS` refers to GWAS Catalog accession GCST90012877 (Schwartzentruber et al., 2021; GRCh38). The scripts document required HapMap3+ and GRCh38 genetic-map references.

Dependencies include bigsnpr, bigstatsr, bigparallelr, bigreadr, data.table and dplyr. The historical environment declaration is provided for context; it is not a tested installation lock for this release.

LD is estimated from the target genotypes. LDpred2-auto does not use clinical phenotype for fitting; scores include APOE. Preserve allele matching, variant order, chain selection and RNG settings.

Raw VCF QC, sample QC, unrelatedness and ancestry inference precede these scripts. Verified clinical reconstruction begins from processed PRS/roster and PHC inputs. This release does not claim a tested restart from raw VCFs. The 26,390-person PRS calculation and full prognostic training were not rerun.

R backing objects can contain machine-specific paths and large matrices. Use a new output directory rather than moving cached pointers. No genotypes, individual scores, fitted weights or reference-data rights are redistributed.
