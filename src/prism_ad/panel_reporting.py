"""Publication-format exports of already executed private panel diagnostics."""
from pathlib import Path
import pandas as pd

def _p(value):
    return r"$<0.001$" if float(value)<.001 else f"{float(value):.4f}"
def _table(headers,rows):
    parts=[r"\begin{longtable}{@{}"+"l"*len(headers)+r"@{}}",r"\toprule",
           " & ".join(headers)+r"\\",r"\midrule\endhead"]
    parts +=[" & ".join(row)+r"\\" for row in rows]
    return "\n".join(parts+[r"\bottomrule\end{longtable}"])

def calibration_report(directory):
    d=pd.read_csv(Path(directory)/"calibration_slope_tests.csv")
    if len(d)!=4:raise ValueError("All four diagnostic slots must be reported")
    rows=[]
    for task in ["CN_MCI","MCI_AD"]:
        for horizon in [3,5]:
            z=d[d.task.eq(task)&d.horizon.eq(horizon)]
            if len(z)!=1:raise ValueError("Calibration keys must be unique")
            r=z.iloc[0]
            count=int(r["n_known"] if "n_known" in r else r["n"])
            rows.append([("CN to first MCI" if task=="CN_MCI" else "MCI to AD")+", "+str(horizon)+" y",
                         str(count),f"{r.slope:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f})",
                         _p(r.p),_p(r.q),str(r.label)])
    text=[r"\clearpage\subsection*{Calibration slope diagnostics}",
          "Slopes were estimated from individual first-eligible held-out forecasts and positive-IPCW known outcomes using weighted logistic diagnostics with a free intercept. The reference slope is 1. Empirical participant-score sandwich covariance used HC1 finite-sample correction. Intervals are nominal normal 95\\% intervals and P values test slope = 1.",
          _table(["Clinical transition, horizon","Known n","Slope (nominal 95\\% CI)","P","BH q","Annotation"],rows),
          "These four slope tests share one new ten-test BH family with the six marginal contrasts in Supplementary Table 4F. A significant slope departure indicates miscalibration of risk spread, not superior prediction. Nonsignificance does not establish calibration across the risk range. Diagnostic fits were not applied to alter forecasts, and uncertainty is conditional on the saved predictions and censoring weights."]
    return "\n\n".join(text)+"\n"

def marginal_report(directory):
    d=pd.read_csv(Path(directory)/"marginal_component_tests.csv")
    if len(d)!=6:raise ValueError("All six marginal comparisons must be reported")
    rows=[]
    for r in d.itertuples():
        rows.append([str(r.component),str(r.group_a)+" versus "+str(r.group_b),
                     str(int(r.n_a))+" / "+str(int(r.n_b)),
                     f"{r.rank_biserial_b_minus_a:.3f}",_p(r.p),_p(r.q),str(r.label)])
    text=[r"\clearpage\subsection*{F. Post hoc marginal contribution comparisons}",
          "All participants were CN at prediction. Groups use all available later clinical diagnoses, rather than a fixed AD horizon. Each individual contributes one value to each component. Comparisons use within-fold midranks and inverse-(pair-specific stratum size + 1) weights, with tie-corrected van Elteren normal inference. No pooling of native contribution scales or adjustment for age, study or follow-up was performed.",
          _table(["Component","First versus second group","n, first / second",r"Rank-biserial $r$","P","BH q","Annotation"],rows),
          "Positive rank-biserial r indicates larger values in the second group, based on weighted within-fold pairwise concordance. It is not a clinical risk difference or a causal effect. The six tests share one new ten-test BH family with four calibration-slope tests in Supplementary Table 3. Ordinary stars denote BH q below 0.05, 0.01 and 0.001. (*) denotes nominal P below 0.05 with q at or above 0.05. All tests are exploratory and conditional on fitted components, with unequal clinical follow-up and overlapping model-training samples."]
    return "\n\n".join(text)+"\n"
