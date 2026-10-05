import pandas as pd
from scipy.stats import wilcoxon

optimal_k = {
    'A': (4, 4, 3),
    'R': (3, 4, 3),
    'N': (5, 4, 3),
    'D': (4, 5, 3),
    'C': (4, 4, 3),
    'Q': (4, 4, 4),
    'E': (4, 5, 4),
    'G': (2, 2, 4),
    'H': (4, 4, 3),
    'I': (3, 8, 8),
    'L': (4, 3, 3),
    'K': (3, 4, 4),
    'M': (4, 4, 4),
    'F': (4, 4, 4),
    'P': (4, 6, 6),
    'S': (6, 5, 4),
    'T': (5, 5, 3),
    'W': (4, 5, 4),
    'Y': (4, 5, 4),
    'V': (4, 8, 9)
}

summary_records = []
for RESIDUE, (k_opt_pvm, k_opt_sbvm, k_opt_ssbvm) in optimal_k.items():
    df_pvm = pd.read_csv(f"data/experiments/product_5x5fold/{RESIDUE}/{RESIDUE}_all_5x5fold_metrics.csv")
    df_sbvm = pd.read_csv(f"data/experiments/sine_5x5fold/{RESIDUE}/{RESIDUE}_all_5x5fold_metrics.csv")
    df_ssbvm = pd.read_csv(f"data/experiments/sine_skewed_5x5fold/{RESIDUE}/{RESIDUE}_all_5x5fold_metrics.csv")

    pvm_k = df_pvm[df_pvm["n_components"] == k_opt_pvm].sort_values(by=["split_idx", "fold_idx"])
    sbvm_k = df_sbvm[df_sbvm["n_components"] == k_opt_sbvm].sort_values(by=["split_idx", "fold_idx"])
    ssbvm_k = df_ssbvm[df_ssbvm["n_components"] == k_opt_ssbvm].sort_values(by=["split_idx", "fold_idx"])

    # Filter each model to its specific optimal component count
    pvm_opt = df_pvm[df_pvm["n_components"] == k_opt_pvm][["split_idx", "fold_idx", "loglikelihood", "aic"]].rename(
        columns={"loglikelihood": "loglik_pvm", "aic": "aic_pvm"}
    )
    sbvm_opt = df_sbvm[df_sbvm["n_components"] == k_opt_sbvm][["split_idx", "fold_idx", "loglikelihood", "aic"]].rename(
        columns={"loglikelihood": "loglik_sbvm", "aic": "aic_sbvm"}
    )
    ssbvm_opt = df_ssbvm[df_ssbvm["n_components"] == k_opt_ssbvm][["split_idx", "fold_idx", "loglikelihood", "aic"]].rename(
        columns={"loglikelihood": "loglik_ssbvm", "aic": "aic_ssbvm"}
    )

    # Merge on (split_idx, fold_idx) to guarantee exact fold-to-fold pairing
    merged = pvm_opt.merge(sbvm_opt, on=["split_idx", "fold_idx"]).merge(ssbvm_opt, on=["split_idx", "fold_idx"])

    # Paired Wilcoxon signed-rank tests across the matched folds
    stat_ssbvm_vs_sbvm, p_ssbvm_vs_sbvm = wilcoxon(merged["loglik_ssbvm"], merged["loglik_sbvm"])
    stat_ssbvm_vs_pvm,  p_ssbvm_vs_pvm  = wilcoxon(merged["loglik_ssbvm"], merged["loglik_pvm"])
    stat_sbvm_vs_pvm,   p_sbvm_vs_pvm   = wilcoxon(merged["loglik_sbvm"], merged["loglik_pvm"])

    def get_significance_marker(p: float) -> str:
        if p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    sig_ssbvm_sbvm = get_significance_marker(p_ssbvm_vs_sbvm)
    sig_ssbvm_pvm  = get_significance_marker(p_ssbvm_vs_pvm)
    sig_sbvm_pvm   = get_significance_marker(p_sbvm_vs_pvm)

    print(f"Paired comparison on {RESIDUE}:")
    print(f"  SSBVM (k={k_opt_ssbvm}) vs SBVM (k={k_opt_sbvm}):  W = {stat_ssbvm_vs_sbvm}, p = {p_ssbvm_vs_sbvm:.4e} ({sig_ssbvm_sbvm})")
    print(f"  SSBVM (k={k_opt_ssbvm}) vs PVM  (k={k_opt_pvm}):   W = {stat_ssbvm_vs_pvm},  p = {p_ssbvm_vs_pvm:.4e} ({sig_ssbvm_pvm})")
    print(f"  SBVM  (k={k_opt_sbvm}) vs PVM  (k={k_opt_pvm}):   W = {stat_sbvm_vs_pvm},   p = {p_sbvm_vs_pvm:.4e} ({sig_sbvm_pvm})")