# =============================================================================
# Step Zero: Prepare GWAS Summary Statistics and Target Genotype Data
# for LDpred2-auto PRS (Alzheimer's Disease)
#
# GWAS summary statistics: Schwartzentruber et al. 2021 (GCST90012877, GRCh38)
# Target genotype data:    QC-processed ADSP NG00067.v17 genotypes restricted
#                          to the inferred European-ancestry cluster
# Internal PLINK prefix:   step12_eur_only (pipeline artifact name only)
# Genetic map:             Eagle v2.4 GRCh38 (genetic_map_hg38_withX.txt.gz)
#                          Loh et al. 2016, Nature Genetics
#                          Source: https://alkesgroup.broadinstitute.org/Eagle/downloads/tables/
# LD reference:            HapMap3+ (Privé et al. 2023, AJHG; figshare doi:10.6084/m9.figshare.21305061)
#
# Output .Rdata contains:
#   df_beta  — sumstats matched to genotype map; _NUM_ID_ indexes into G_sub/Gipt
#   G_sub    — genotype matrix subset to matched variants only (FBM, ~618K cols)
#   Gipt     — mean-imputed version of G_sub (FBM, same dims)
#   POS2     — genetic positions (cM) for matched variants only, same order as df_beta
#   fam      — sample table from obj_bigSNP$fam (FID, IID, etc.)
#   n_hm3    — total HapMap3+ variant count for correct ld_size in Step 1a LDSC
#
# Scientific caveats:
#   - Schwartzentruber 2021 sumstats combine a GWAX (proxy phenotype) with a
#     direct case-control GWAS. GWAX designs have known biases including
#     survival bias and non-random parental illness reporting.
#   - n_eff is hardcoded from the direct GWAS component only (21,982 cases /
#     41,944 controls). Automated SE-based estimate fails for GWAX+GWAS designs.
#   - snp_match is performed by position (join_by_pos = TRUE) to handle rsID
#     inconsistencies between the sumstats and the ADSP genotype map.
#   - Genetic positions use linear interpolation (snp_asGeneticPos2) against
#     the Eagle v2.4 GRCh38 map. More accurate than nearest-neighbor.
#
# Run from the project root through scripts/03_run_all_sample_prs.sh after
# populating config/project.env. Paths are supplied only through environment
# variables; phenotype is not loaded in this step.
# =============================================================================

# Library paths — must be first, before any library() calls
if (nzchar(Sys.getenv("PRS_R_LIB")))
  .libPaths(c(Sys.getenv("PRS_R_LIB"), .libPaths()))

library(bigsnpr)
library(bigstatsr)
library(bigparallelr)
library(data.table)
library(dplyr)

# --- BLAS parallelism protection ---------------------------------------------
# Applied globally here to cover all bigsnpr/bigstatsr parallel calls
# (snp_fastImputeSimple, snp_asGeneticPos2, big_copy, etc.).
# Without this, any ncores > 1 call can trigger "two levels of parallelism".
options(bigstatsr.check.parallel.blas = FALSE)
tryCatch(
  bigparallelr::set_blas_ncores(1),
  error = function(e) message("Note: set_blas_ncores unavailable: ", e$message)
)

# --- Logging helpers ---------------------------------------------------------

log_step <- function(...) {
  cat(paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", paste(...), "\n"))
  flush.console()
}

step_time <- function(t0) {
  elapsed <- as.numeric(proc.time() - t0, units = "secs")[3]
  log_step(sprintf("  Step time: %.1f min", elapsed / 60))
}

# --- Paths -------------------------------------------------------------------

HM3_MAP_FILE    <- Sys.getenv("HM3_MAP_FILE", unset="")
SUMSTATS        <- Sys.getenv("SUMMARY_STATS", unset="")
BED_FILE        <- Sys.getenv("FINAL_BFILE", unset="")
if (nzchar(BED_FILE) && !endsWith(BED_FILE, ".bed")) BED_FILE <- paste0(BED_FILE, ".bed")
GENETIC_MAP_DIR <- Sys.getenv("GENETIC_MAP_DIR", unset="")
# NOTE: genetic_map_hg38_withX.txt.gz must already exist in GENETIC_MAP_DIR.
# download_genetic_map() will find and decompress it if the .txt is not yet present.
# If the cluster has no outbound internet, pre-decompress manually:
#   gunzip -k /path/to/reference_data/genetic_map_hg38_withX.txt.gz
OUT_DIR         <- Sys.getenv("PRS_OUTPUT_DIR", unset="")
OUTPUT_RDATA    <- file.path(OUT_DIR, "ADSP_AD_prepared.Rdata")
NCORES          <- as.integer(Sys.getenv("PRS_NCORES", "16"))

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PIPELINE_START <- proc.time()
log_step("================================================")
log_step("Step Zero: PRS Data Preparation — Alzheimer's Disease")
log_step("Started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
log_step("Output: ", OUTPUT_RDATA)
log_step("================================================")

# =============================================================================
# Step 1: Load HapMap3+ variant map
#
# Used for: (a) filtering sumstats to well-imputed common variants,
#           (b) recording n_hm3 for correct ld_size in Step 1a LDSC.
# =============================================================================

log_step("Step 1: Loading HapMap3+ variant map...")
t0 <- proc.time()

map_hm3 <- readRDS(HM3_MAP_FILE)
n_hm3   <- nrow(map_hm3)   # saved to .Rdata — required by Step 1a snp_ldsc
log_step("  HapMap3+ variants (n_hm3):", n_hm3)
step_time(t0)

# =============================================================================
# Step 2: Load and format GWAS summary statistics
#
# Column mapping (GCST90012877_buildGRCh38.txt):
#   seqnames -> chr, start -> pos, variant_id -> rsid,
#   effect_allele -> a1, other_allele -> a0,
#   beta -> beta, standard_error -> beta_se, p_value -> pvalue
#
# n_eff is hardcoded from the direct GWAS component only — see caveats above.
# =============================================================================

SUMSTATS_CACHE <- file.path(OUT_DIR, "sumstats_formatted.rds")

if (file.exists(SUMSTATS_CACHE)) {
  log_step("Step 2: Loading cached summary statistics...")
  t0 <- proc.time()
  sumstats <- readRDS(SUMSTATS_CACHE)
  log_step("  Loaded", nrow(sumstats), "variants from cache")
  if (!"n_eff" %in% names(sumstats))
    stop("ERROR: cached sumstats lacks n_eff — delete ", SUMSTATS_CACHE, " and rerun.")
  step_time(t0)
} else {
  log_step("Step 2: Loading and formatting GWAS summary statistics...")
  t0 <- proc.time()

  sumstats_raw <- bigreadr::fread2(SUMSTATS)

  if (all(c("chromosome", "base_pair_location", "variant_id", "effect_allele", "other_allele", "beta", "standard_error", "p_value") %in% names(sumstats_raw))) {
    log_step("  Detected GWAS Catalog GRCh38 harmonised schema")
    sumstats_raw <- sumstats_raw %>% dplyr::transmute(
      seqnames = as.character(chromosome),
      start = as.integer(base_pair_location),
      variant_id = variant_id,
      effect_allele = effect_allele,
      other_allele = other_allele,
      beta = beta,
      standard_error = standard_error,
      p_value = p_value
    )
  }

  sumstats <- sumstats_raw %>%
    mutate(
      chr     = as.integer(gsub("chr", "", seqnames)),
      pos     = as.integer(start),
      rsid    = variant_id,
      a1      = effect_allele,
      a0      = other_allele,
      beta    = beta,
      beta_se = standard_error,
      pvalue  = p_value
    ) %>%
    select(chr, pos, rsid, a1, a0, beta, beta_se, pvalue) %>%
    filter(!is.na(chr), chr %in% 1:22,
           !is.na(beta), !is.na(beta_se), beta_se > 0)

  if (any(sumstats$pos == 0, na.rm = TRUE))
    stop("ERROR: position value of 0 detected — `start` column appears 0-based. ",
         "Fix: pos = as.integer(start) + 1")

  # Hardcoded n_eff from direct GWAS component (Schwartzentruber 2021)
  n_cases    <- 21982
  n_controls <- 41944
  sumstats$n_eff <- 4 / (1/n_cases + 1/n_controls)
  log_step("  n_eff (direct GWAS, Schwartzentruber 2021):", round(sumstats$n_eff[1]))

  saveRDS(sumstats, SUMSTATS_CACHE)
  log_step("  Loaded", nrow(sumstats), "autosomal variants — cached")
  step_time(t0)
}

# =============================================================================
# Step 3: Filter summary statistics to HapMap3+ variants
# =============================================================================

log_step("Step 3: Filtering sumstats to HapMap3+ variants...")
t0 <- proc.time()

sumstats_hm3 <- sumstats[sumstats$rsid %in% map_hm3$rsid, ]
log_step("  Retained:", nrow(sumstats_hm3), "of", nrow(sumstats),
         "(", round(100 * nrow(sumstats_hm3) / nrow(sumstats), 1), "%)")
step_time(t0)

# =============================================================================
# Step 4: Read QC-processed target genotype data
# =============================================================================

log_step("Step 4: Reading QC-processed ADSP genotype data...")
t0 <- proc.time()

RDS_FILE <- sub("\\.bed$", ".rds", BED_FILE)
BK_FILE  <- sub("\\.bed$", ".bk",  BED_FILE)

if (!file.exists(RDS_FILE) || !file.exists(BK_FILE)) {
  log_step("  Converting PLINK bed -> bigSNP format (one-time, may take ~30 min)...")
  snp_readBed(BED_FILE)
} else {
  log_step("  bigSNP .rds/.bk already exist, skipping conversion")
}

obj_bigSNP <- snp_attach(RDS_FILE)
G   <- obj_bigSNP$genotypes
CHR <- obj_bigSNP$map$chromosome
POS <- obj_bigSNP$map$physical.pos
fam <- obj_bigSNP$fam   # saved to .Rdata for FID/IID in Step 1a output

log_step("  Samples:", nrow(fam))
log_step("  Variants:", ncol(G))
step_time(t0)

# =============================================================================
# Step 5: Get genetic positions for ALL variants (full map)
#
# Uses snp_asGeneticPos2() with linear interpolation against Eagle v2.4
# GRCh38 map (genetic_map_hg38_withX.txt.gz). More accurate than the
# nearest-neighbor snp_asGeneticPos(). download_genetic_map() will find
# the already-downloaded file in GENETIC_MAP_DIR and use it directly.
#
# POS2_full covers all variants in G — it is subsetted to matched variants
# in Step 7, after snp_match has identified which columns are used.
# =============================================================================

log_step("Step 5: Calculating genetic positions for all variants (hg38, linear interp)...")
t0 <- proc.time()

genetic_map <- download_genetic_map(type   = "hg38_price",
                                    dir    = GENETIC_MAP_DIR,
                                    ncores = NCORES)

POS2_full <- snp_asGeneticPos2(as.integer(CHR), POS,
                                genetic_map = genetic_map)

log_step("  Genetic positions computed:", length(POS2_full), "variants")
log_step("  Range:", round(min(POS2_full), 4), "to", round(max(POS2_full), 4), "cM")
step_time(t0)

# =============================================================================
# Step 6: Match summary statistics to target genotype map
#
# snp_match returns df_beta with _NUM_ID_ indexing into the FULL G matrix
# (62M columns). This index is used in Step 7 to subset G before imputation,
# then RESET to sequential 1:nrow(df_beta) so Step 1a indexes into G_sub.
# =============================================================================

DF_BETA_CACHE <- file.path(OUT_DIR, "df_beta_matched.rds")

if (file.exists(DF_BETA_CACHE)) {
  log_step("Step 6: Loading cached matched sumstats...")
  t0 <- proc.time()
  df_beta <- readRDS(DF_BETA_CACHE)
  log_step("  Loaded", nrow(df_beta), "variants from cache")
  if (!"n_eff" %in% names(df_beta))
    stop("ERROR: cached df_beta lacks n_eff. Delete ", DF_BETA_CACHE, " and rerun.")
  step_time(t0)
} else {
  log_step("Step 6: Matching summary statistics to genotype map...")
  t0 <- proc.time()

  map_target <- setNames(
    obj_bigSNP$map[, c("chromosome", "marker.ID", "physical.pos", "allele1", "allele2")],
    c("chr", "rsid", "pos", "a1", "a0")
  )
  map_target$chr <- as.integer(gsub("chr", "", map_target$chr))

  df_beta <- snp_match(sumstats_hm3, map_target, join_by_pos = TRUE)

  log_step("  Variants matched:", nrow(df_beta))
  log_step("  Variants lost:", nrow(sumstats_hm3) - nrow(df_beta))

  if (nrow(df_beta) == 0)
    stop("ERROR: No variants matched. Check chromosome format in .bim file.")

  saveRDS(df_beta, DF_BETA_CACHE)
  log_step("  Cached to:", DF_BETA_CACHE)
  step_time(t0)
}

# =============================================================================
# Step 7: Subset G to matched variants, then impute
#
# CRITICAL ORDER: match first, subset second, impute third.
# Imputing all 62M variants before subsetting would require ~1.6 TB disk
# and many hours. Subsetting to ~618K first reduces imputation to ~16 GB.
#
# After big_copy, G_sub has columns 1..nrow(df_beta) corresponding to the
# matched variants IN THE SAME ORDER as df_beta rows. _NUM_ID_ is then reset
# to sequential 1:nrow(df_beta) so Step 1a can index directly into G_sub.
#
# POS2 is also subsetted here from POS2_full using the original _NUM_ID_
# before it is overwritten.
# =============================================================================

G_SUB_RDS <- file.path(OUT_DIR, "G_sub.rds")
G_SUB_BK  <- file.path(OUT_DIR, "G_sub.bk")

if (file.exists(G_SUB_BK) && !file.exists(G_SUB_RDS)) {
  log_step("Step 7: Partial G_sub found (.bk without .rds) — removing and rebuilding...")
  file.remove(G_SUB_BK)
}

if (file.exists(G_SUB_RDS) && file.exists(G_SUB_BK)) {
  log_step("Step 7: G_sub already exists, loading...")
  t0 <- proc.time()
  G_sub <- big_attach(G_SUB_RDS)
  log_step("  Loaded:", ncol(G_sub), "variants x", nrow(G_sub), "samples")

  # Recover the original _NUM_ID_ mapping to re-derive POS2 and Gipt if needed
  # (If cache was built correctly, df_beta _NUM_ID_ is already sequential)
  if (max(df_beta[["_NUM_ID_"]]) > ncol(G_sub)) {
    log_step("  _NUM_ID_ still points to full G — resetting to sequential...")
    # This means the cache was loaded but _NUM_ID_ was not reset in a previous run.
    # Re-derive POS2 from the subset order recorded in the cache file name.
    stop("ERROR: G_sub exists but _NUM_ID_ is not sequential. ",
         "Delete df_beta_matched.rds AND G_sub.rds/.bk and rerun from Step 6.")
  }
  step_time(t0)
} else {
  log_step("Step 7: Subsetting to", nrow(df_beta), "matched variants...")
  t0 <- proc.time()

  # Capture original _NUM_ID_ BEFORE resetting — needed for POS2 subsetting
  original_num_id <- df_beta[["_NUM_ID_"]]

  G_sub <- big_copy(G, ind.col = original_num_id,
                    backingfile = file.path(OUT_DIR, "G_sub"))
  log_step("  Subset complete:", ncol(G_sub), "variants x", nrow(G_sub), "samples")

  # Subset POS2 to matched variants in the same order as df_beta rows
  POS2 <- POS2_full[original_num_id]
  log_step("  POS2 subsetted:", length(POS2), "positions")

  # Reset _NUM_ID_ to sequential so Step 1a indexes into G_sub not full G
  df_beta[["_NUM_ID_"]] <- seq_len(nrow(df_beta))
  log_step("  _NUM_ID_ reset to sequential 1:", nrow(df_beta))

  log_step("  Imputing missing genotypes on subset...")
  G_sub <- snp_fastImputeSimple(G_sub, method = "mean2", ncores = NCORES)
  log_step("  Imputation complete")

  # Update the df_beta cache with the reset _NUM_ID_
  saveRDS(df_beta, DF_BETA_CACHE)
  log_step("  Updated df_beta cache with sequential _NUM_ID_")
  step_time(t0)
}

# If loaded from G_sub cache, POS2 may not be in memory yet — derive it.
# The .Rdata save below will capture the correct POS2 either way.
if (!exists("POS2")) {
  log_step("  POS2 not yet in memory — deriving from POS2_full and df_beta _NUM_ID_...")
  # At this point _NUM_ID_ is sequential (1:nrow) after a previous correct run.
  # Reconstruct POS2 from the bim physical positions via genetic map.
  bim_pos_subset <- POS[obj_bigSNP$map$chromosome %in% df_beta$chr]
  stop("ERROR: POS2 is missing and cannot be reliably reconstructed from cache. ",
       "Delete G_sub.rds/.bk and df_beta_matched.rds and rerun from Step 6.")
}

# Rename G_sub to Gipt for .Rdata compatibility with Step 1a
# G_sub IS the imputed matrix (snp_fastImputeSimple modifies in place).
# We expose it as both G_sub (raw subset, pre-imputation reference) and
# Gipt (imputed). Since snp_fastImputeSimple works in place, they are the
# same object — Step 1a uses Gipt for LD and PRS, G_sub/G for LDSC LD.
# To give Step 1a both, we save G_sub as both G and Gipt in the .Rdata.
# Step 1a's snp_cor(G, ...) and snp_cor(Gipt, ...) will both work correctly.
Gipt <- G_sub

# =============================================================================
# Step 8: Save all prepared objects to .Rdata for Step 1a
#
# Objects saved:
#   df_beta — matched sumstats; _NUM_ID_ is sequential 1:nrow, indexes G_sub/Gipt
#   G_sub   — genotype subset (FBM pointer; .bk file is the actual data on disk)
#   Gipt    — same object as G_sub (mean-imputed in place); Step 1a uses this
#             for both LD computation and PRS scoring
#   POS2    — genetic positions (cM) for matched variants, same order as df_beta
#   fam     — sample table (FID, IID, sex, phenotype) for output labeling
#   n_hm3   — total HapMap3+ variant count for snp_ldsc ld_size in Step 1a
# =============================================================================

log_step("Step 8: Saving prepared objects to .Rdata...")
t0 <- proc.time()

save(df_beta, G_sub, Gipt, POS2, fam, n_hm3, file = OUTPUT_RDATA)

log_step("  Saved to:", OUTPUT_RDATA)
log_step("  df_beta:  ", nrow(df_beta), "variants | _NUM_ID_ sequential: 1 to", nrow(df_beta))
log_step("  G_sub/Gipt:", nrow(G_sub), "samples x", ncol(G_sub), "variants")
log_step("  POS2:     ", length(POS2), "positions |",
         round(min(POS2), 2), "to", round(max(POS2), 2), "cM")
log_step("  fam:      ", nrow(fam), "samples")
log_step("  n_hm3:    ", n_hm3)
step_time(t0)

# =============================================================================
# Summary
# =============================================================================

elapsed <- as.numeric(proc.time() - PIPELINE_START, units = "secs")[3]
log_step("================================================")
log_step("Step Zero complete — ready for Step 1a")
log_step("  Total wall time:     ", sprintf("%.1f min", elapsed / 60))
log_step("  Samples:             ", nrow(fam))
log_step("  Variants in df_beta: ", nrow(df_beta))
log_step("  Output .Rdata:       ", OUTPUT_RDATA)
log_step("================================================")
log_step("Next: update Step 1a input_rdata to:")
log_step("  ", OUTPUT_RDATA)
