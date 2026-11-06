"""Figures built from saved rerun outputs (model comparison, confusion matrix, SHAP bar) + supplementary workbook."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from results import load, PRETTY, MODELS, OUT

plt.rcParams.update({"savefig.dpi": 300, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
FD = os.path.join(OUT, "figures")


def main():
    R = load(); b = R["boot"]; sel = R["sel"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    names = MODELS
    for a, met, title in zip(ax, ["macro_f1", "qwk"], ["Macro-F1", "Quadratic-weighted kappa"]):
        pts = [b[(b.model == m) & (b.metric == met)].iloc[0] for m in names]
        y = np.arange(len(names))
        a.errorbar([p.point for p in pts], y, xerr=[[p.point - p.ci_lo for p in pts], [p.ci_hi - p.point for p in pts]], fmt="o",
                   color="#2E5E8C", ecolor="#8AA9C8", capsize=3)
        a.scatter([1.0], [len(names)], marker="D", color="#C0392B", zorder=3)
        a.set_yticks(list(y) + [len(names)]); a.set_yticklabels([PRETTY[m] + (" *" if m == sel else "") for m in names] + ["Recovered rule"])
        a.set_xlabel(title + " (held-out test, 95% bootstrap CI)"); a.grid(axis="x", alpha=0.3)
    ax[0].set_title("* selected by cross-validation", fontsize=9, loc="left")
    plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_model_comparison.png")); plt.close()
    cm = np.array(R["cm"][sel])
    fig, ax = plt.subplots(figsize=(4.2, 3.6)); im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["Low", "Medium", "High"]); ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{PRETTY[sel]} (tuned), test n = 1,500", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_confusion_selected.png")); plt.close()
    imp = R["imp"].mean_abs_shap_all_classes.sort_values()
    fig, ax = plt.subplots(figsize=(6, 5)); rule = ["Soil_Moisture", "Rainfall_mm", "Crop_Growth_Stage", "Temperature_C", "Wind_Speed_kmh", "Mulching_Used"]
    ax.barh(imp.index, imp.values, color=["#C0392B" if k in rule else "#9AA5B1" for k in imp.index])
    ax.set_xlabel("Mean |SHAP| (summed over one-hot columns, averaged over classes)"); ax.set_title("Global SHAP importance; red = recovered rule features", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_shap_global.png")); plt.close()
    # supplementary workbook
    T = lambda f: pd.read_csv(os.path.join(OUT, "tables", f))
    sheets = {"S1_audit_marginals": T("S1_audit_marginals.csv"), "S2_univariate_tests": T("S2_univariate_tests_corrected.csv"),
              "S3_rule_score_vs_label": pd.read_csv(os.path.join(OUT, "tables", "S3_rule_score_vs_label.csv")),
              "S3b_shallow_trees": T("S4_shallow_tree_rule_recovery.csv"),
              "S4_per_class_all_models": T("C4_test_per_class_all_models.csv"), "S5_test_bootstrap_CI": T("C5_test_bootstrap_CI.csv"),
              "S5b_test_metrics_all": T("C3_test_metrics_all_models.csv"), "S5c_calibration": T("C6_calibration.csv"),
              "S6_importance_and_tau": R["imp"].reset_index().rename(columns={"index": "feature"}),
              "S6b_kendall_tau": R["tau"].reset_index(), "S7_cv5_fold_scores": T("C1_cv5_fold_scores_default_and_tuned.csv"),
              "S7b_repeated_cv_scores": T("C7_repeated_cv_5x10_fold_scores.csv"), "S7c_pairwise_tests": T("C8_pairwise_tests_holm.csv"),
              "S7e_partition_stability": T("C10_partition_stability.csv"),
              "S8_scenarios": T("M0_scenarios_150.csv"), "S8b_representativeness": T("M0_scenario_representativeness.csv"),
              "S8c_scenario_results": T("M1_scenario_results_exact.csv"), "S9_bounds": T("M5_bounds_sensitivity.csv"),
              "S10_moea_runs": T("M2_moea_runs_all_seeds.csv"), "S10b_moea_tests": T("M4_moea_tests_holm.csv"),
              "S11_joint_sensitivity": T("M7_joint_sensitivity_indices.csv"), "S12_common_protocol_bench": T("M8_common_protocol_benchmark.csv"),
              "S13_hyperparameters": pd.DataFrame([{"model": m, **{k: str(v) for k, v in p.items()}} for m, p in R["bp"]["best_params"].items()])}
    dest = os.path.join(OUT, "Supplementary_Tables_S1-S13.xlsx")
    with pd.ExcelWriter(dest) as w:
        for k, d in sheets.items():
            d.to_excel(w, sheet_name=k[:31], index=False)
    print("figs + supplementary written")


if __name__ == "__main__":
    main()
