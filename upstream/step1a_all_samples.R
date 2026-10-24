Sys.umask("0077")
# =============================================================================
# Step 1a: LDpred2-auto PRS Computation — Alzheimer's Disease
#
# Loads prepared data from Step Zero (.Rdata) and computes PRS using
# LDpred2-auto with LD estimated from the QC-processed ADSP target genotypes,
# restricted to the inferred European-ancestry cluster.
#
# Differences from external-LD approach (original Step Zero):
#   - LD computed via snp_cor() on the processed target genotypes
#     (not HapMap3+ precomputed)
#   - Two LD matrices built: corr (from Gipt, for LDpred2-auto),
#     corr_o (from G_sub, for LDSC h2 estimation)
#   - LDSC ld_size equals the number of variants represented by its LD scores.
#     n_hm3 is the original map size and is not an LDSC denominator.
#
# Scientific caveats:
#   - snp_cor size=3/1000 means LD window = 3 cM. Standard for LDpred2.
#   - compact=FALSE in as_SFBM: compact SFBM silently causes NA chains in
#     some bigsnpr versions. Always use standard SFBM here.
#   - Chain quality filtered on both h2 and p (3 SD from median), not just
#     range of corr_est, which can fail when many chains diverge.
#   - burn_in=500, num_iter=500 hardcoded for reproducibility across bigsnpr
#     versions whose defaults may differ.
#   - BLAS parallelism explicitly disabled to prevent "two levels of
#     parallelism" error when NCORES Gibbs chains run concurrently.
#
# BUG FIXES vs original:
#   [1] SFBM .sbk backing files now built directly in OUT_DIR (not TMP_DIR).
#       Original built in TMP_DIR but checked OUT_DIR for cache, so the cache
#       branch was NEVER entered — LD recomputed every run (~6 hrs each time).
#   [2] big_prodVec now passes ind.col = df_beta[["_NUM_ID_"]]. Original would
#       error when any variants were removed by the Step 2 variance filter,
#       because beta_auto length < ncol(Gipt).
#   [3] Cache branch now loads ld_scores_o.rds directly instead of recomputing
#       ld_o from corr_o SFBM via Matrix::colSums block extraction (slow,
#       fragile, not supported cleanly across bigsnpr versions).
#   [4] ld_scores_o.rds included in cache validity check. Script now errors
#       clearly if .rds exists but ld_scores_o.rds was lost (partial run),
#       rather than silently falling into the broken SFBM recompute path.
#
# Run from the project root through scripts/03_run_all_sample_prs.sh after
# populating config/project.env. Paths are supplied only through environment
# variables; phenotype is not loaded in this step.
# =============================================================================

# Library paths — must be first
if (nzchar(Sys.getenv("PRS_R_LIB")))
  .libPaths(c(Sys.getenv("PRS_R_LIB"), .libPaths()))

library(bigsnpr)
library(bigstatsr)
library(bigparallelr)
library(data.table)
library(dplyr)

# --- BLAS parallelism protection ---------------------------------------------
# Applied globally here to cover all bigsnpr/bigstatsr parallel calls.
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

INPUT_RDATA    <- file.path(Sys.getenv("PRS_OUTPUT_DIR", unset=""), "ADSP_AD_prepared.Rdata")
# TMP_DIR kept for truly temporary intermediate files only (not SFBM backing).
TMP_DIR        <- file.path(Sys.getenv("PRS_OUTPUT_DIR", unset=""), "tmp")
OUT_DIR         <- Sys.getenv("PRS_OUTPUT_DIR", unset="")
OUT_PREFIX     <- file.path(OUT_DIR, "ADSP_AD_PRS_step1a_all_samples")
NCORES         <- as.integer(Sys.getenv("PRS_NCORES", "40"))


required_env <- c("PRS_OUTPUT_DIR")
if (any(!nzchar(Sys.getenv(required_env))))
  stop("Set PRS_OUTPUT_DIR to a new external private directory before running.")
argv_file <- sub("^--file=", "", commandArgs(FALSE)[grepl("^--file=",commandArgs(FALSE))])
if (length(argv_file)==1L) {
  code_root <- dirname(dirname(normalizePath(argv_file,mustWork=TRUE)))
  forbidden <- code_root
  if (basename(dirname(code_root)) %in% c("PRISM-AD_Public","PRISM-AD_Local"))
    forbidden <- c(forbidden,dirname(code_root))
  candidate <- normalizePath(OUT_DIR,mustWork=FALSE)
  if (any(vapply(forbidden,function(x)candidate==x || startsWith(candidate,paste0(x,"/")),logical(1))))
    stop("PRS outputs must remain outside the public checkout and distribution packages.")
}

if ((file.exists(file.path(OUT_DIR,"ldsc_results.rds")) ||
     file.exists(file.path(OUT_DIR,"multi_auto_step1a.rds"))) &&
    !file.exists(file.path(OUT_DIR,"PRS_parameter_fingerprint.rds")))
  stop("Unregistered historical LDSC/auto caches detected. Preserve them and choose a new private output.")

dir.create(TMP_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PIPELINE_START <- proc.time()
log_step("================================================")
log_step("Step 1a: LDpred2-auto PRS — Alzheimer's Disease")
log_step("Started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
log_step("Input:  ", INPUT_RDATA)
log_step("Output: ", OUT_PREFIX)
log_step("================================================")

# =============================================================================
# Step 1: Load prepared objects from Step Zero
#
# Loads: df_beta, G_sub, Gipt, POS2, fam, n_hm3
#   df_beta — matched sumstats; _NUM_ID_ is sequential, indexes G_sub/Gipt
#   G_sub   — raw genotype subset (FBM)
#   Gipt    — mean-imputed genotype subset (FBM, same object as G_sub)
#   POS2    — genetic positions (cM) for matched variants, same order as df_beta
#   fam     — sample table for FID/IID output labeling
#   n_hm3   — original HapMap3+ map size for provenance only
# =============================================================================

log_step("Step 1: Loading prepared data from Step Zero...")
t0 <- proc.time()

if (!file.exists(INPUT_RDATA))
  stop("ERROR: Input .Rdata not found: ", INPUT_RDATA,
       "\nRun Step Zero first to generate prepared data.")

load(INPUT_RDATA)
Gipt$is_read_only <- TRUE
G_sub$is_read_only <- TRUE
stopifnot(Gipt$is_read_only,G_sub$is_read_only)

# Validate all required objects are present
required <- c("df_beta", "G_sub", "Gipt", "POS2", "fam", "n_hm3")
missing  <- required[!sapply(required, exists)]
if (length(missing) > 0)
  stop("ERROR: Missing objects in .Rdata: ", paste(missing, collapse=", "),
       "\nDelete ", INPUT_RDATA, " and rerun Step Zero.")

# Validate _NUM_ID_ is sequential (indexes G_sub, not full G)
if (max(df_beta[["_NUM_ID_"]]) > ncol(G_sub))
  stop("ERROR: _NUM_ID_ max (", max(df_beta[["_NUM_ID_"]]), ") exceeds G_sub columns (",
       ncol(G_sub), "). Step Zero did not reset _NUM_ID_ after subsetting. ",
       "Delete ", INPUT_RDATA, " and rerun Step Zero.")

# Validate POS2 length matches df_beta
if (length(POS2) != nrow(df_beta))
  stop("ERROR: POS2 length (", length(POS2), ") != df_beta rows (", nrow(df_beta), "). ",
       "Step Zero POS2 subsetting is misaligned. Rerun Step Zero.")

log_step("  df_beta: ", nrow(df_beta), "variants")
log_step("  G_sub:   ", nrow(G_sub), "samples x", ncol(G_sub), "variants")
log_step("  n_hm3:   ", n_hm3)
log_step("  n_eff:   ", round(df_beta$n_eff[1]))

# Log chromosome coverage for sanity check
chr_counts <- table(df_beta$chr)
log_step("  Chr coverage (variants per chr):")
for (ch in names(chr_counts))
  log_step("    chr", ch, ":", chr_counts[[ch]], "variants")

step_time(t0)

# =============================================================================
# Step 2: Filter zero-variance variants before LD computation
#
# snp_cor returns NaN for variants with zero variance (monomorphic or
# near-monomorphic after QC). NaN propagates into ld_o and causes snp_ldsc
# to fail. Filtering before LD computation prevents this entirely.
#
# After filtering, df_beta$_NUM_ID_ holds the original column indices into
# G_sub/Gipt for the retained variants. snp_cor uses these via ind.col.
# POS2 is also filtered to match df_beta in length and order.
# big_prodVec in Step 7 also uses these indices via ind.col — required
# because ncol(Gipt) == original variant count, not filtered count.
# =============================================================================

log_step("Step 2: Filtering zero-variance variants...")
t0 <- proc.time()

# big_colstats operates on all ncol(G_sub) columns
col_stats <- big_colstats(G_sub, ncores = NCORES)

# Categorize every variant's removal reason for per-variant logging
var_is_nan  <- is.nan(col_stats$var)
var_is_na   <- is.na(col_stats$var) & !is.nan(col_stats$var)  # NA but not NaN
var_is_inf  <- is.infinite(col_stats$var)
var_is_zero <- !is.nan(col_stats$var) & !is.na(col_stats$var) &
               !is.infinite(col_stats$var) & col_stats$var == 0

log_step("  Variance summary (pre-filter):")
log_step("    n variants total:   ", length(col_stats$var))
log_step("    n var == 0:         ", sum(var_is_zero),
         "(monomorphic after imputation — all alleles identical in cohort)")
log_step("    n var is NaN:       ", sum(var_is_nan),
         "(snp_cor would return NaN — propagates to ld_o and breaks snp_ldsc)")
log_step("    n var is NA:        ", sum(var_is_na),
         "(big_colstats failed to compute — likely all-missing column in G_sub)")
log_step("    n var is Inf:       ", sum(var_is_inf),
         "(numerical overflow — should not occur with normal genotype data)")

good_var  <- which(is.finite(col_stats$var) & col_stats$var > 0)
bad_var   <- setdiff(seq_along(col_stats$var), good_var)
n_removed <- length(bad_var)

# Per-variant removal log — written to a file and echoed to console
REMOVED_VARIANTS_FILE <- paste0(OUT_PREFIX, "_removed_variants.txt")

if (n_removed > 0) {
  log_step("  Removing", n_removed, "variants — writing per-variant log to:")
  log_step("  ", REMOVED_VARIANTS_FILE)

  # Assign removal reason per variant
  reason <- character(length(bad_var))
  for (i in seq_along(bad_var)) {
    idx <- bad_var[i]
    if (var_is_nan[idx])  reason[i] <- "NaN_variance"
    else if (var_is_na[idx])   reason[i] <- "NA_variance"
    else if (var_is_inf[idx])  reason[i] <- "Inf_variance"
    else if (var_is_zero[idx]) reason[i] <- "zero_variance_monomorphic"
    else                       reason[i] <- "other_nonpositive_variance"
  }

  # Build removal table — join with df_beta for variant identity fields
  removed_df <- data.frame(
    G_sub_col_idx = bad_var,
    chr           = df_beta$chr[bad_var],
    pos           = df_beta$pos[bad_var],
    rsid          = df_beta$rsid[bad_var],
    a1            = df_beta$a1[bad_var],
    a0            = df_beta$a0[bad_var],
    variance      = col_stats$var[bad_var],
    mean_af       = col_stats$sum[bad_var] / (2 * nrow(G_sub)),  # approx allele freq
    reason        = reason,
    stringsAsFactors = FALSE
  )

  write.table(removed_df, file = REMOVED_VARIANTS_FILE,
              sep = "\t", row.names = FALSE, quote = FALSE)

  # Console summary by reason and chromosome
  reason_tbl <- table(removed_df$reason)
  log_step("  Removal reasons:")
  for (r in names(reason_tbl))
    log_step("    ", r, ":", reason_tbl[[r]])

  chr_removed_tbl <- table(removed_df$chr)
  log_step("  Removed variants by chromosome:")
  for (ch in names(chr_removed_tbl))
    log_step("    chr", ch, ":", chr_removed_tbl[[ch]], "removed")

  # Log first few removed variants for spot-checking
  n_preview <- min(10, nrow(removed_df))
  log_step("  First", n_preview, "removed variants (for spot check):")
  for (i in seq_len(n_preview)) {
    log_step("    [", i, "] chr", removed_df$chr[i], ":", removed_df$pos[i],
             removed_df$rsid[i], "| var =", signif(removed_df$variance[i], 4),
             "| reason:", removed_df$reason[i])
  }

  # Apply filter
  df_beta <- df_beta[good_var, ]
  POS2    <- POS2[good_var]
  # BUG FIX [2]: _NUM_ID_ must hold original G_sub column indices for the
  # retained variants so that big_prodVec (Step 7) can pass ind.col correctly.
  # Without this, big_prodVec(Gipt, beta_auto) fails: length(beta_auto) !=
  # ncol(Gipt) when any variants are removed here.
  df_beta[["_NUM_ID_"]] <- good_var

  log_step("  Variants retained:", nrow(df_beta))
  chr_counts_post <- table(df_beta$chr)
  log_step("  Chr coverage post-filter (variants per chr):")
  for (ch in names(chr_counts_post))
    log_step("    chr", ch, ":", chr_counts_post[[ch]], "variants")
} else {
  log_step("  All", ncol(G_sub), "variants passed variance check — no removal log written")
}

# Guard: if all variants were removed the pipeline cannot continue
if (nrow(df_beta) == 0)
  stop("ERROR: All variants removed by variance filter in Step 2. ",
       "No variants remain for LD computation. ",
       "Check that G_sub was correctly imputed in Step Zero and that the ",
       "input VCFs passed variant-level QC (MAF, call rate filters).")

log_step("  Post-filter variant count check passed:", nrow(df_beta), "variants remain")
step_time(t0)

# =============================================================================
# Step 3: Compute LD matrices chromosome by chromosome
#
# Two LD matrices are built:
#   corr   — from Gipt (imputed): used by snp_ldpred2_auto
#   corr_o — from G_sub (raw):    used by snp_ldsc for h2 estimation
#
# BUG FIX [1]: SFBM backing files (.sbk) are now built directly in OUT_DIR,
# not TMP_DIR. The original code built .sbk in TMP_DIR but set SFBM_BK to
# OUT_DIR — so SFBM_BK never existed and the cache branch was never entered,
# forcing a full LD recompute on every run (~6 hours each time).
#
# BUG FIX [3] + [4]: ld_scores_o.rds is now part of the cache validity check
# and is loaded directly in the cache branch, instead of recomputing ld_o from
# corr_o via Matrix::colSums block extraction (slow, fragile across bigsnpr
# versions, and incorrect when SFBM block-diagonal indexing misbehaves).
#
# compact=FALSE: compact SFBM silently causes all-NA chains in some bigsnpr
# versions. Always use standard SFBM regardless of disk space savings.
# =============================================================================

# BUG FIX [1]: All four cache files now live in OUT_DIR (not a split between
# TMP_DIR for .sbk and OUT_DIR for .rds).
SFBM_CACHE      <- file.path(OUT_DIR, "ld_corr_sfbm.rds")
SFBM_BK         <- file.path(OUT_DIR, "ld_corr_sfbm.sbk")
SFBM_O_CACHE    <- file.path(OUT_DIR, "ld_corr_o_sfbm.rds")
SFBM_O_BK       <- file.path(OUT_DIR, "ld_corr_o_sfbm.sbk")
LD_SCORES_CACHE <- file.path(OUT_DIR, "ld_scores_o.rds")    # BUG FIX [4]
# Defined here (not in Steps 4/5) so the cascade stale-removal block below can
# evict all downstream caches when LD must be recomputed.
LDSC_CACHE       <- file.path(OUT_DIR, "ldsc_results.rds")
MULTI_AUTO_CACHE <- file.path(OUT_DIR, "multi_auto_step1a.rds")

# Backing file prefixes for as_SFBM — in OUT_DIR so .sbk co-locates with .rds
sfbm_corr_prefix   <- file.path(OUT_DIR, "ld_corr_sfbm")
sfbm_corr_o_prefix <- file.path(OUT_DIR, "ld_corr_o_sfbm")

# Remove partial files from any previous crashed run
# Check both .rds and .sbk for each matrix — partial means one exists without other
if (file.exists(SFBM_BK)   && !file.exists(SFBM_CACHE))   { file.remove(SFBM_BK);        log_step("  Removed partial corr .sbk (no matching .rds)") }
if (file.exists(SFBM_CACHE) && !file.exists(SFBM_BK))     { file.remove(SFBM_CACHE);     log_step("  Removed orphaned corr .rds (no matching .sbk)") }
if (file.exists(SFBM_O_BK) && !file.exists(SFBM_O_CACHE)) { file.remove(SFBM_O_BK);      log_step("  Removed partial corr_o .sbk (no matching .rds)") }
if (file.exists(SFBM_O_CACHE) && !file.exists(SFBM_O_BK)) { file.remove(SFBM_O_CACHE);   log_step("  Removed orphaned corr_o .rds (no matching .sbk)") }

# BUG FIX [4]: ld_scores_o.rds is now part of cache validity check
cache_complete <- (
  file.exists(SFBM_CACHE)     && file.exists(SFBM_BK) &&
  file.exists(SFBM_O_CACHE)   && file.exists(SFBM_O_BK) &&
  file.exists(LD_SCORES_CACHE)
)

# If cache is incomplete for ANY reason, remove ALL five LD cache files before
# entering the compute branch. Without this, a run where only ld_scores_o.rds
# is missing would enter the compute branch with both .sbk files still on disk,
# causing as_SFBM() to error or silently corrupt the existing backing files when
# it tries to create new ones at the same paths.
if (!cache_complete) {
  stale <- c(SFBM_CACHE, SFBM_BK, SFBM_O_CACHE, SFBM_O_BK, LD_SCORES_CACHE,
             LDSC_CACHE, MULTI_AUTO_CACHE)
  stale_exist <- stale[file.exists(stale)]
  if (length(stale_exist) > 0) {
    log_step("  Incomplete LD cache — removing stale LD + downstream files before recompute.")
    log_step("  (LDSC and multi_auto are downstream of LD and must be recomputed with it.)")
    for (f in stale_exist) log_step("    removing:", f)
    file.remove(stale_exist)
  }
}

if (cache_complete) {

  log_step("Step 3: Loading cached LD matrices and ld scores...")
  t0 <- proc.time()

  corr   <- readRDS(SFBM_CACHE)
  corr_o <- readRDS(SFBM_O_CACHE)

  log_step("  corr   loaded: ", nrow(corr), "x", ncol(corr), "(SFBM)")
  log_step("  corr_o loaded: ", nrow(corr_o), "x", ncol(corr_o), "(SFBM)")

  # Validate loaded dimensions match current df_beta
  if (nrow(corr) != nrow(df_beta))
    stop("ERROR: Cached corr rows (", nrow(corr), ") != df_beta rows (",
         nrow(df_beta), "). Variance filtering may have changed the variant set. ",
         "Delete ld_corr_sfbm.* and rerun.")

  # BUG FIX [3]: Load ld_o directly from saved RDS instead of recomputing
  # from corr_o. The original recomputation used Matrix::colSums on SFBM
  # block-diagonal slices, which is slow, unreliable across bigsnpr versions,
  # and unnecessary since we save ld_scores_o.rds at compute time.
  ld_o <- readRDS(LD_SCORES_CACHE)
  log_step("  ld_o loaded:   ", length(ld_o), "scores (from ld_scores_o.rds)")

  if (length(ld_o) != nrow(df_beta))
    stop("ERROR: Cached ld_o length (", length(ld_o), ") != df_beta rows (",
         nrow(df_beta), "). Delete ld_scores_o.rds and rerun LD computation.")

  log_step("  ld_o range:    ", round(min(ld_o), 2), "to", round(max(ld_o), 2))
  log_step("  ld_o non-finite:", sum(!is.finite(ld_o)))

  step_time(t0)

} else {

  # Log which cache files are missing so it's clear why we're recomputing
  log_step("Step 3: Cache incomplete — computing LD matrices from target genotypes...")
  if (!file.exists(SFBM_CACHE))       log_step("  Missing: ", SFBM_CACHE)
  if (!file.exists(SFBM_BK))         log_step("  Missing: ", SFBM_BK)
  if (!file.exists(SFBM_O_CACHE))     log_step("  Missing: ", SFBM_O_CACHE)
  if (!file.exists(SFBM_O_BK))       log_step("  Missing: ", SFBM_O_BK)
  if (!file.exists(LD_SCORES_CACHE))  log_step("  Missing: ", LD_SCORES_CACHE)
  log_step("  This is the most time-consuming step (~2-6 hours for 26K samples)")

  t0 <- proc.time()

  ld_o <- c()
  first_chr <- TRUE

  for (chr in 1:22) {
    log_step("  Chr", chr, "...")
    ind_chr <- which(df_beta$chr == chr)
    if (length(ind_chr) == 0) {
      log_step("    No matched variants, skipping")
      next
    }

    ind_chr_geno <- df_beta[["_NUM_ID_"]][ind_chr]

    log_step("    Variants:", length(ind_chr),
             "| G_sub col range:", min(ind_chr_geno), "-", max(ind_chr_geno),
             "| cM range:", round(min(POS2[ind_chr]), 2), "-", round(max(POS2[ind_chr]), 2))

    # LD from imputed genotypes — for LDpred2-auto
    corr0 <- snp_cor(
      Gipt,
      ind.col   = ind_chr_geno,
      size      = 3 / 1000,       # 3 cM window
      infos.pos = POS2[ind_chr],
      ncores    = NCORES
    )

    # LD from raw genotypes — for LDSC h2 estimation
    corr0_o <- snp_cor(
      G_sub,
      ind.col   = ind_chr_geno,
      size      = 3 / 1000,
      infos.pos = POS2[ind_chr],
      ncores    = NCORES
    )

    log_step("    snp_cor done | corr0 nnz:", length(corr0@x),
             "| corr0_o nnz:", length(corr0_o@x))

    chr_ld_o <- Matrix::colSums(corr0_o^2)
    n_nan_ld  <- sum(is.nan(chr_ld_o))
    if (n_nan_ld > 0)
      log_step("    WARNING:", n_nan_ld, "NaN ld scores on chr", chr,
               "— these variants likely have zero variance in G_sub despite passing Gipt filter")

    if (first_chr) {
      ld_o   <- chr_ld_o
      # BUG FIX [1]: backingfile = sfbm_corr_prefix (OUT_DIR), not TMP_DIR.
      # compact=FALSE: prevents all-NA chain bug in some bigsnpr versions.
      corr   <- as_SFBM(corr0,   backingfile = sfbm_corr_prefix,   compact = FALSE)
      corr_o <- as_SFBM(corr0_o, backingfile = sfbm_corr_o_prefix, compact = FALSE)
      first_chr <- FALSE
    } else {
      ld_o   <- c(ld_o, chr_ld_o)
      corr$add_columns(corr0,   nrow(corr))
      corr_o$add_columns(corr0_o, nrow(corr_o))
    }

    log_step("    Chr", chr, "done | running ld_o length:", length(ld_o),
             "| SFBM dims:", nrow(corr), "x", ncol(corr),
             sprintf("| %.1f min elapsed", as.numeric(proc.time() - t0)[3] / 60))
  }

  # Validate LD matrix dimensions match df_beta
  if (nrow(corr) != nrow(df_beta))
    stop("ERROR: LD matrix rows (", nrow(corr), ") != df_beta rows (",
         nrow(df_beta), "). Chromosome loop produced misaligned blocks.")

  if (length(ld_o) != nrow(df_beta))
    stop("ERROR: ld_o length (", length(ld_o), ") != df_beta rows (", nrow(df_beta),
         "). Chromosome loop accumulated ld_o incorrectly.")

  log_step("  LD matrices complete:", nrow(corr), "x", ncol(corr))
  log_step("  ld_o complete:", length(ld_o), "scores")
  log_step("  ld_o non-finite:", sum(!is.finite(ld_o)))

  # Cache SFBM objects and ld_o scores
  # BUG FIX [1]: .sbk files now already live in OUT_DIR (from as_SFBM above).
  # saveRDS writes only the small R object metadata — the .sbk is already there.
  saveRDS(corr,   SFBM_CACHE)
  saveRDS(corr_o, SFBM_O_CACHE)
  saveRDS(ld_o,   LD_SCORES_CACHE)
  log_step("  Cached corr (.rds + .sbk), corr_o (.rds + .sbk), ld_scores_o.rds to OUT_DIR")
  log_step("  SFBM backing files at:", sfbm_corr_prefix, ".sbk /", sfbm_corr_o_prefix, ".sbk")

  step_time(t0)
}

# Re-estimate after the final variance filter, on the supplied meta-analysis scale.
stopifnot(nrow(df_beta)==length(ld_o), nrow(corr)==length(ld_o),
          ncol(corr)==length(ld_o), all(is.finite(df_beta$beta_se)), all(df_beta$beta_se>0))
df_beta$n_eff <- unname(quantile(8 / df_beta$beta_se^2, .999, type=7))
# Cached LDSC/auto objects from another denominator or N must never be reused.
fingerprint <- list(version="matched_LD_SE_N_v1", N_eff=df_beta$n_eff[1],
                    ld_size=length(ld_o), beta=df_beta$beta, beta_se=df_beta$beta_se,
                    variant_ids=df_beta[["_NUM_ID_"]], ld_scores=ld_o)
fingerprint_file <- file.path(OUT_DIR,"PRS_parameter_fingerprint.rds")
if (file.exists(LDSC_CACHE) || file.exists(MULTI_AUTO_CACHE)) {
  if (!file.exists(fingerprint_file) ||
      !identical(readRDS(fingerprint_file),fingerprint))
    stop("Existing LDSC/auto caches have no matching parameter fingerprint. Use a new PRS_OUTPUT_DIR; preserve historical caches.")
}
saveRDS(fingerprint,fingerprint_file)

# =============================================================================
# Step 4: Estimate heritability via LD Score Regression
#
# ld_size is the matched LD reference size, equal to length(ld_o).
# Using matched-variant count underestimates genome-wide LD and biases h2.
# =============================================================================

if (file.exists(LDSC_CACHE)) {
  log_step("Step 4: Loading cached LDSC results...")
  t0 <- proc.time()
  ldsc    <- readRDS(LDSC_CACHE)
  h2_init <- max(ldsc[["h2"]], 1e-4)
  log_step("  h2:", round(h2_init, 4), "| intercept:", round(ldsc[["int"]], 4))
  step_time(t0)
} else {
  log_step("Step 4: Running LD Score Regression...")
  t0 <- proc.time()

  chi2 <- (df_beta$beta / df_beta$beta_se)^2

  # Guard: drop any residual NaN/Inf in ld_o before passing to snp_ldsc
  ldsc_ok <- is.finite(ld_o) & is.finite(chi2)
  n_excluded <- sum(!ldsc_ok)
  if (n_excluded > 0)
    log_step("  WARNING:", n_excluded, "variants with non-finite ld_o or chi2 excluded from LDSC",
             "\n    ld_o non-finite:", sum(!is.finite(ld_o)),
             "\n    chi2 non-finite:", sum(!is.finite(chi2)))

  log_step("  LDSC input: n_variants =", sum(ldsc_ok),
           "| ld_size (matched LD) =", length(ld_o),
           "| sample_size =", round(df_beta$n_eff[1]))
  log_step("  chi2 summary: mean =", round(mean(chi2[ldsc_ok]), 3),
           "| median =", round(median(chi2[ldsc_ok]), 3),
           "| max =", round(max(chi2[ldsc_ok]), 1))

  ldsc <- snp_ldsc(
    ld_score    = ld_o[ldsc_ok],
    ld_size     = length(ld_o),   # same variant universe as the LD-score sums
    chi2        = chi2[ldsc_ok],
    sample_size = df_beta$n_eff[1]
  )

  h2_init <- max(ldsc[["h2"]], 1e-4)
  log_step("  h2 estimate:", round(h2_init, 4))
  log_step("  intercept:  ", round(ldsc[["int"]], 4),
           "(>1.1 suggests inflation — check GWAX calibration)")
  if (ldsc[["int"]] > 1.1)
    log_step("  WARNING: intercept > 1.1 indicates potential population stratification",
             "or summary stat inflation. Review QC before trusting PRS.")
  if (h2_init <= 1e-4)
    log_step("  WARNING: h2 <= 0 — LDSC returned non-positive heritability.",
             "h2_init floored to 1e-4 for LDpred2. Check ld_o, chi2, and n_eff.")

  saveRDS(ldsc, LDSC_CACHE)
  step_time(t0)
}

# =============================================================================
# Step 5: Run LDpred2-auto
#
# burn_in=500, num_iter=500 hardcoded for reproducibility — bigsnpr defaults
# may change across versions.
# allow_jump_sign=FALSE: prevents sign flips in Gibbs chains that indicate
# divergence. Recommended for production PRS.
# set.seed(116): matches original pipeline for reproducibility.
# =============================================================================

if (file.exists(MULTI_AUTO_CACHE)) {
  log_step("Step 5: Loading cached LDpred2-auto results...")
  t0 <- proc.time()
  multi_auto <- readRDS(MULTI_AUTO_CACHE)
  log_step("  Loaded", length(multi_auto), "chains from cache")
  step_time(t0)
} else {
  log_step("Step 5: Running LDpred2-auto (", NCORES, "chains, ~1-3 hrs)...")
  log_step("  h2_init:       ", round(h2_init, 4))
  log_step("  vec_p_init:    30 values from 1e-4 to 0.2 (log scale)")
  log_step("  burn_in:       500")
  log_step("  num_iter:      500")
  log_step("  allow_jump_sign: FALSE")
  log_step("  shrink_corr:   0.95")
  log_step("  NCORES:        ", NCORES)
  t0 <- proc.time()

  set.seed(116)   # for reproducibility

  multi_auto <- snp_ldpred2_auto(
    corr            = corr,
    df_beta         = df_beta,
    h2_init         = h2_init,
    vec_p_init      = seq_log(1e-4, 0.2, length.out = 30),
    burn_in         = 500,
    num_iter        = 500,
    report_step     = 20,
    ncores          = NCORES,
    allow_jump_sign = FALSE,
    shrink_corr     = 0.95
  )

  saveRDS(multi_auto, MULTI_AUTO_CACHE)
  log_step("  Gibbs results cached to:", MULTI_AUTO_CACHE)
  step_time(t0)
}

# =============================================================================
# Step 6: Filter diverged chains and extract final beta estimates
#
# Filter on BOTH h2 and p (3 SD from median) — more robust than range of
# corr_est alone, which can produce a collapsed threshold when many chains
# diverge simultaneously.
# =============================================================================

log_step("Step 6: Filtering chains and extracting beta estimates...")
t0 <- proc.time()

range_h2 <- sapply(multi_auto, function(x) x$h2_est)
range_p  <- sapply(multi_auto, function(x) x$p_est)

n_na_h2 <- sum(is.na(range_h2))
n_na_p  <- sum(is.na(range_p))
if (n_na_h2 > 0 || n_na_p > 0)
  log_step("  WARNING:", n_na_h2, "chains returned NA h2,", n_na_p,
           "chains returned NA p — Gibbs sampler failed on those chains")

log_step("  Chain h2 summary: median =", round(median(range_h2, na.rm=TRUE), 4),
         "| SD =", round(sd(range_h2, na.rm=TRUE), 4),
         "| range:", round(min(range_h2, na.rm=TRUE), 4), "-", round(max(range_h2, na.rm=TRUE), 4))
log_step("  Chain p  summary: median =", round(median(range_p, na.rm=TRUE), 6),
         "| SD =", round(sd(range_p, na.rm=TRUE), 6),
         "| range:", round(min(range_p, na.rm=TRUE), 6), "-", round(max(range_p, na.rm=TRUE), 6))
log_step("  Chain h2 (first 10):", paste(round(head(range_h2, 10), 4), collapse=", "))
log_step("  Chain p  (first 10):", paste(round(head(range_p,  10), 6), collapse=", "))

# Keep chains within 3 SD of median on both h2 and p
# max(..., 1e-6) prevents false divergence when all chains converge (sd=0)
keep <- (
  abs(range_h2 - median(range_h2, na.rm=TRUE)) < max(3 * sd(range_h2, na.rm=TRUE), 1e-6) &
  abs(range_p  - median(range_p,  na.rm=TRUE)) < max(3 * sd(range_p,  na.rm=TRUE), 1e-6)
)
keep[is.na(keep)] <- FALSE

if (sum(keep) == 0) {
  warning("All chains flagged as diverged — using all non-NA chains. Check chain diagnostics.")
  keep <- !is.na(range_h2) & !is.na(range_p)
  if (sum(keep) == 0) {
    warning("All chains returned NA — using all chains regardless.")
    keep <- rep(TRUE, length(keep))
  }
}

log_step("  Chains kept:", sum(keep), "of", length(keep),
         "(", round(100 * sum(keep)/length(keep), 1), "%)")
log_step("  Final h2:", round(median(range_h2[keep]), 4))
log_step("  Final p: ", round(median(range_p[keep]),  6))

# as.matrix() ensures rowMeans works even when sum(keep) == 1.
# sapply() returns a plain vector (not a matrix) when only one chain is kept,
# and rowMeans() errors on a vector: "must be array of at least two dimensions".
betas_mat <- as.matrix(sapply(multi_auto[keep], function(x) x$beta_est))
beta_auto  <- rowMeans(betas_mat)
log_step("  beta_auto length:", length(beta_auto))
log_step("  Non-zero betas:  ", sum(beta_auto != 0),
         "(", round(100 * mean(beta_auto != 0), 2), "%)")
log_step("  beta_auto range: ", round(min(beta_auto), 6), "to", round(max(beta_auto), 6))

step_time(t0)

# =============================================================================
# Step 7: Compute PRS for each ADSP sample
#
# BUG FIX [2]: big_prodVec now uses ind.col = df_beta[["_NUM_ID_"]] to
# restrict multiplication to the retained variant columns in Gipt. Without
# this, big_prodVec requires length(beta_auto) == ncol(Gipt). After Step 2
# variance filtering removes any variants, nrow(df_beta) < ncol(Gipt) and
# the call errors. ind.col maps each beta to its correct Gipt column.
#
# ncores=1: avoids residual BLAS conflict; n_samples x n_variants completes
# in seconds regardless.
# =============================================================================

log_step("Step 7: Computing polygenic risk scores...")
t0 <- proc.time()

log_step("  Gipt:      ", nrow(Gipt), "samples x", ncol(Gipt), "variants (total FBM)")
log_step("  beta_auto: ", length(beta_auto), "variants")
log_step("  ind.col:   ", length(df_beta[["_NUM_ID_"]]), "indices (", min(df_beta[["_NUM_ID_"]]),
         "to", max(df_beta[["_NUM_ID_"]]), ")")

# Sanity check: beta_auto length must match ind.col length
if (length(beta_auto) != length(df_beta[["_NUM_ID_"]]))
  stop("ERROR: beta_auto length (", length(beta_auto), ") != ind.col length (",
       length(df_beta[["_NUM_ID_"]]), "). Chain filtering or _NUM_ID_ is misaligned.")

# BUG FIX [2]: pass ind.col so big_prodVec uses only the retained variant columns
prs <- big_prodVec(Gipt, beta_auto, ind.col = df_beta[["_NUM_ID_"]], ncores = 1)

log_step("  PRS computed for", length(prs), "samples")
log_step("  Mean:", round(mean(prs), 4), "| SD:", round(sd(prs), 4))
log_step("  Min: ", round(min(prs),  4), "| Max:", round(max(prs), 4))
log_step("  Skewness check: n > mean+3SD =", sum(prs > mean(prs) + 3*sd(prs)),
         "| n < mean-3SD =", sum(prs < mean(prs) - 3*sd(prs)))

# Guard: PRS vector must have exactly one entry per sample in fam
# Mismatch here means Gipt and G_sub have diverged from fam since Step Zero —
# e.g. if fam was reloaded from a different object than Gipt was attached from.
if (length(prs) != nrow(fam))
  stop("ERROR: PRS length (", length(prs), ") != fam rows (", nrow(fam), "). ",
       "Gipt and fam are misaligned — they must both come from the same ",
       "snp_attach() call in Step Zero. Delete ", INPUT_RDATA, " and rerun.")

step_time(t0)

# =============================================================================
# Step 8: Save outputs
#
# Three output files:
#   *_scores.txt       — FID, IID, PRS (one row per sample)
#   *_betas.txt        — per-variant LDpred2 beta estimates
#   *_chain_diag.txt   — per-chain h2, p, kept status for QC
# =============================================================================

log_step("Step 8: Saving outputs...")
t0 <- proc.time()

# PRS scores with sample IDs from fam
prs_out <- data.frame(
  FID = fam$family.ID,
  IID = fam$sample.ID,
  PRS = prs
)
prs_file <- paste0(OUT_PREFIX, "_scores.txt")
write.table(prs_out, file = prs_file,
            sep = "\t", row.names = FALSE, quote = FALSE)
log_step("  PRS scores:  ", prs_file, "(", nrow(prs_out), "rows)")

# Per-variant beta estimates
# Normalize p-value column name: bigsnpr sumstats use "p" internally;
# some pipelines rename it to "pvalue". Handle both so select() never errors.
if ("p" %in% names(df_beta) && !"pvalue" %in% names(df_beta)) {
  df_beta <- df_beta %>% rename(pvalue = p)
  log_step("  NOTE: Renamed df_beta column 'p' -> 'pvalue' for output consistency")
} else if (!"pvalue" %in% names(df_beta)) {
  log_step("  NOTE: Neither 'p' nor 'pvalue' found in df_beta — pvalue column omitted from betas output")
}

beta_out <- df_beta %>%
  # chr, pos, a1, a0, beta, beta_se are always present (required by snp_match).
  # rsid, n_eff, pvalue are standard but not guaranteed — use any_of() so a
  # missing column is silently skipped rather than throwing a dplyr error.
  select(chr, pos, any_of(c("rsid", "a2")), a1, a0, beta, beta_se,
         any_of(c("pvalue", "n_eff"))) %>%
  mutate(beta_ldpred2 = beta_auto)
beta_file <- paste0(OUT_PREFIX, "_betas.txt")
write.table(beta_out, file = beta_file,
            sep = "\t", row.names = FALSE, quote = FALSE)
log_step("  Beta estimates:", beta_file, "(", nrow(beta_out), "variants)")

# Chain diagnostics
chain_diag <- data.frame(
  chain  = seq_along(multi_auto),
  h2_est = range_h2,
  p_est  = range_p,
  kept   = keep
)
diag_file <- paste0(OUT_PREFIX, "_chain_diag.txt")
write.table(chain_diag, file = diag_file,
            sep = "\t", row.names = FALSE, quote = FALSE)
log_step("  Chain diagnostics:", diag_file)

step_time(t0)

# =============================================================================
# Step 9: Note on SFBM backing files
#
# BUG FIX [1]: SFBM .sbk files are now in OUT_DIR (not TMP_DIR) and are
# needed on subsequent runs for cache loading. Do NOT delete them. TMP_DIR
# cleanup is kept in case any true temp files accumulate, but the .sbk pattern
# now only hits files in TMP_DIR (none should exist after this fix).
# =============================================================================

log_step("Step 9: Checking for orphaned temp files in TMP_DIR...")
tmp_sbk_files <- list.files(TMP_DIR, pattern = "\\.sbk$", full.names = TRUE)
if (length(tmp_sbk_files) > 0) {
  log_step("  Found", length(tmp_sbk_files), "orphaned .sbk in TMP_DIR — removing:")
  for (f in tmp_sbk_files) log_step("    ", f)
  file.remove(tmp_sbk_files)
} else {
  log_step("  TMP_DIR clean — no orphaned .sbk files")
}
log_step("  SFBM backing files retained at OUT_DIR (required for cache on next run):")
log_step("    ", SFBM_BK)
log_step("    ", SFBM_O_BK)

# =============================================================================
# Summary
# =============================================================================

elapsed <- as.numeric(proc.time() - PIPELINE_START, units = "secs")[3]
log_step("================================================")
log_step("Step 1a complete.")
log_step("  Total wall time:   ", sprintf("%.1f min", elapsed / 60))
log_step("  Samples scored:    ", nrow(prs_out))
log_step("  Variants used:     ", sum(beta_auto != 0))
log_step("  Final h2:          ", round(median(range_h2[keep]), 4))
log_step("  Final p:           ", round(median(range_p[keep]),  6))
log_step("  PRS scores:        ", prs_file)
log_step("  Beta estimates:    ", beta_file)
log_step("  Chain diagnostics: ", diag_file)
log_step("  SFBM cache:        ", SFBM_CACHE, "+", SFBM_BK)
log_step("  SFBM_o cache:      ", SFBM_O_CACHE, "+", SFBM_O_BK)
log_step("  LD scores cache:   ", LD_SCORES_CACHE)
log_step("================================================")
